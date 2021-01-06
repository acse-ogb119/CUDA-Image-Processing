#include <iostream>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include <nppi_data_exchange_and_initialization.h>
#include <opencv2/imgcodecs.hpp>

#include "../include/utils.h"
#include "../include/loadSaveImage.h"

static const int numBins = 1024;

__global__ void rgb_to_xyY_kernel(float *d_r, float *d_g, float *d_b,
                                  float *d_x, float *d_y, float *d_log_Y,
                                  float delta, int num_pixels_y, int num_pixels_x)
{
    int ny = num_pixels_y;
    int nx = num_pixels_x;
    int2 image_index_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
    int image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

    if (image_index_2d.x < nx && image_index_2d.y < ny)
    {
        float r = d_r[image_index_1d];
        float g = d_g[image_index_1d];
        float b = d_b[image_index_1d];

        float X = (r * 0.4124f) + (g * 0.3576f) + (b * 0.1805f);
        float Y = (r * 0.2126f) + (g * 0.7152f) + (b * 0.0722f);
        float Z = (r * 0.0193f) + (g * 0.1192f) + (b * 0.9505f);

        float L = X + Y + Z;
        float x = X / L;
        float y = Y / L;

        float log_Y = log10f(delta + Y);

        d_x[image_index_1d] = x;
        d_y[image_index_1d] = y;
        d_log_Y[image_index_1d] = log_Y;
    }
}

__global__ void normalize_cdf(const unsigned int *const d_input_cdf, float *const d_output_cdf, int n)
{
    float normalization_constant = 1.f / d_input_cdf[n - 1];
    int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (global_index_1d < n)
    {
        int input_value = d_input_cdf[global_index_1d];
        float output_value = input_value * normalization_constant;
        d_output_cdf[global_index_1d] = output_value;
    }
}

__global__ void tonemap_kernel(const float *const d_x, const float *const d_y,
                               const float *const d_log_Y, const float *const d_cdf_norm,
                               float *const d_r_new, float *const d_g_new, float *const d_b_new,
                               const float min_log_Y, const float max_log_Y, const float log_Y_range,
                               const int num_bins, const int num_pixels_y, const int num_pixels_x)
{
    int ny = num_pixels_y;
    int nx = num_pixels_x;
    int2 image_index_2d = make_int2((blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y);
    int image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

    if (image_index_2d.x < nx && image_index_2d.y < ny)
    {
        float x = d_x[image_index_1d];
        float y = d_y[image_index_1d];
        float log_Y = d_log_Y[image_index_1d];
        int bin_index = min(num_bins - 1, int((num_bins * (log_Y - min_log_Y)) / log_Y_range));
        float Y_new = d_cdf_norm[bin_index];

        float X_new = x * (Y_new / y);
        float Z_new = (1 - x - y) * (Y_new / y);

        float r_new = (X_new * 3.2406f) + (Y_new * -1.5372f) + (Z_new * -0.4986f);
        float g_new = (X_new * -0.9689f) + (Y_new * 1.8758f) + (Z_new * 0.0415f);
        float b_new = (X_new * 0.0557f) + (Y_new * -0.2040f) + (Z_new * 1.0570f);

        d_r_new[image_index_1d] = r_new;
        d_g_new[image_index_1d] = g_new;
        d_b_new[image_index_1d] = b_new;
    }
}

void preProcess(thrust::device_vector<float> &imgData,
                thrust::device_vector<float> &imgHDR,
                thrust::device_vector<float> &d_x,
                thrust::device_vector<float> &d_y,
                thrust::device_vector<float> &d_logLuminance,
                thrust::device_vector<unsigned int> &d_cdf,
                size_t &rows, size_t &cols,
                const std::string &filename)
{
    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    // allocate and load input image
    loadImageHDR(filename, imgData, rows, cols);
    // allocate output image
    size_t numPixels = rows * cols;
    imgHDR.resize(numPixels * 3);
    d_x.resize(numPixels);
    d_y.resize(numPixels);
    d_logLuminance.resize(numPixels);
    d_cdf.assign(numBins, 0);

    //first thing to do is split incoming BGR float data into separate channels
    thrust::device_vector<float> d_red(numPixels), d_green(numPixels), d_blue(numPixels);

    //split the image channels
    // thrust::device_vector<float> imgData_SoA(numPixels * 3);
    // auto zipIterator = thrust::make_zip_iterator(thrust::make_tuple(blue.begin(), green.begin(), red.begin()));
    // thrust::scatter(imgData.begin(), imgData.end(), ???, zipIterator);
    const float *pSrc = thrust::raw_pointer_cast(imgData.data());
    int nSrcStep = 3 * cols * sizeof(float);
    float *const aDst[3] = {thrust::raw_pointer_cast(d_blue.data()),
                            thrust::raw_pointer_cast(d_green.data()),
                            thrust::raw_pointer_cast(d_red.data())};
    int nDstStep = cols * sizeof(float);
    NppiSize oSizeROI{.width = cols, .height = rows};
    nppiCopy_32f_C3P3R(pSrc, nSrcStep, aDst, nDstStep, oSizeROI);

    //convert from RGB space to chrominance/luminance space xyY
    dim3 blockSize(32, 16, 1);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y, 1);
    rgb_to_xyY_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_red.data()),
                                               thrust::raw_pointer_cast(d_green.data()),
                                               thrust::raw_pointer_cast(d_blue.data()),
                                               thrust::raw_pointer_cast(d_x.data()),
                                               thrust::raw_pointer_cast(d_y.data()),
                                               thrust::raw_pointer_cast(d_logLuminance.data()),
                                               .0001f, rows, cols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void postProcess(const std::string &output_file,
                 thrust::device_vector<float> &imageHDR,
                 const thrust::device_vector<float> &d_logLuminance,
                 const thrust::device_vector<float> &d_x,
                 const thrust::device_vector<float> &d_y,
                 const thrust::device_vector<unsigned int> &d_cdf,
                 const int numRows, const int numCols,
                 const float min_log_Y, const float max_log_Y)
{
    //first normalize the cdf to a maximum value of 1
    //this is how we compress the range of the luminance channel
    thrust::device_vector<float> d_cdf_normalized(numBins);

    int numThreads = 192;
    int numBlocks = (numBins + numThreads - 1) / numThreads;
    normalize_cdf<<<numBlocks, numThreads>>>(thrust::raw_pointer_cast(d_cdf.data()),
                                             thrust::raw_pointer_cast(d_cdf_normalized.data()),
                                             numBins);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    //allocate memory for the output RGB channels
    size_t numPixels = numRows * numCols;
    thrust::device_vector<float> d_red(numPixels), d_green(numPixels), d_blue(numPixels);

    float log_Y_range = max_log_Y - min_log_Y;

    dim3 blockSize(32, 16, 1);
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
                  (numRows + blockSize.y - 1) / blockSize.y);
    //map each luminance value to its new value and then transform back to RGB space
    //TODO: could really use a typedef to avoid all this thrust::raw_pointer_cast(d_vector.data()) repitition
    tonemap_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_x.data()),
                                            thrust::raw_pointer_cast(d_y.data()),
                                            thrust::raw_pointer_cast(d_logLuminance.data()),
                                            thrust::raw_pointer_cast(d_cdf_normalized.data()),
                                            thrust::raw_pointer_cast(d_red.data()),
                                            thrust::raw_pointer_cast(d_green.data()),
                                            thrust::raw_pointer_cast(d_blue.data()),
                                            min_log_Y, max_log_Y, log_Y_range,
                                            numBins, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    //recombine the image channels
    const float *const aSrc[3] = {thrust::raw_pointer_cast(d_blue.data()),
                                  thrust::raw_pointer_cast(d_green.data()),
                                  thrust::raw_pointer_cast(d_red.data())};
    int nSrcStep = numCols * sizeof(float);
    float *pDst = thrust::raw_pointer_cast(imageHDR.data());
    int nDstStep = 3 * numCols * sizeof(float);
    NppiSize oSizeROI{.width = numCols, .height = numRows};
    nppiCopy_32f_P3C3R(aSrc, nSrcStep, pDst, nDstStep, oSizeROI);

    saveImageHDR(imageHDR, numRows, numCols, output_file);
}

__global__ void simple_histo_kernel(const float *const d_logLuminance, unsigned int *const d_bins,
                                    float min_logLum, float logLum_range,
                                    const int numBins, const int count)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < count)
    {
        float lum_i = d_logLuminance[tid];
        int myBin = (lum_i - min_logLum) / logLum_range * numBins;
        atomicAdd(&(d_bins[myBin]), 1);
    }
}

// TODO: this function should take thrust::device_vector 's in directly and do the reduce
// and exclusive scan on them
// void cuda_histogram_and_prefixsum(const float *const d_logLuminance,
//                                   unsigned int *const d_cdf,
//                                   float &min_logLum, float &max_logLum,
//                                   const int numRows, const int numCols)
// {
//     int numPixels = numRows * numCols;

//     // 1) find the minimum and maximum value in the input logLuminance channel
//     //    store in min_logLum and max_logLum
//     thrust::device_ptr<const float> dev_ptr_logLuminance(d_logLuminance);
//     max_logLum = thrust::reduce(thrust::device, dev_ptr_logLuminance, dev_ptr_logLuminance + (size_t)numPixels, max_logLum, thrust::maximum<float>());
//     min_logLum = thrust::reduce(thrust::device, dev_ptr_logLuminance, dev_ptr_logLuminance + (size_t)numPixels, min_logLum, thrust::minimum<float>());
//     std::cout << max_logLum << " " << min_logLum << std::endl;

//     // 2) subtract them to find the range
//     float logLum_range = max_logLum - min_logLum;

//     // 3) generate a histogram of all the values in the logLuminance channel using
//     //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
//     int NumThreadsPerBlock = 512;
//     int NumBlocks = (numPixels + NumThreadsPerBlock - 1) / NumThreadsPerBlock;
//     dim3 blockSize(NumThreadsPerBlock, 1, 1);
//     dim3 gridSize(NumBlocks, 1, 1);
//     simple_histo_kernel<<<gridSize, blockSize>>>(d_logLuminance, d_cdf, min_logLum, logLum_range, numBins, numPixels);

//     // 4) Perform an exclusive scan (prefix sum) on the histogram to get
//     //    the cumulative distribution of luminance values
//     thrust::device_ptr<unsigned int> dev_ptr_cdf(d_cdf);
//     thrust::exclusive_scan(thrust::device, dev_ptr_cdf, dev_ptr_cdf + numBins, dev_ptr_cdf);
// }
void cuda_histogram_and_prefixsum(const thrust::device_vector<float> &d_logLuminance,
                                  thrust::device_vector<unsigned int> &d_cdf,
                                  float &min_logLum, float &max_logLum,
                                  const int numRows, const int numCols)
{
    int numPixels = numRows * numCols;

    // 1) find the minimum and maximum value in the input logLuminance channel
    //    store in min_logLum and max_logLum
    max_logLum = thrust::reduce(d_logLuminance.begin(), d_logLuminance.end(), max_logLum, thrust::maximum<float>());
    min_logLum = thrust::reduce(d_logLuminance.begin(), d_logLuminance.end(), min_logLum, thrust::minimum<float>());
    std::cout << max_logLum << " " << min_logLum << std::endl;

    // 2) subtract them to find the range
    float logLum_range = max_logLum - min_logLum;

    // 3) generate a histogram of all the values in the logLuminance channel using
    //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    int NumThreadsPerBlock = 512;
    int NumBlocks = (numPixels + NumThreadsPerBlock - 1) / NumThreadsPerBlock;
    dim3 blockSize(NumThreadsPerBlock, 1, 1);
    dim3 gridSize(NumBlocks, 1, 1);
    simple_histo_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_logLuminance.data()),
                                                 thrust::raw_pointer_cast(d_cdf.data()),
                                                 min_logLum, logLum_range, numBins, numPixels);

    // 4) Perform an exclusive scan (prefix sum) on the histogram to get
    //    the cumulative distribution of luminance values
    thrust::exclusive_scan(d_cdf.begin(), d_cdf.end(), d_cdf.begin());
}

void tonemap_HDR(const std::string &input_file, const std::string &output_file)
{
    size_t numRows, numCols;

    thrust::device_vector<float> imgData, imgHDR;
    thrust::device_vector<float> d_x, d_y;
    thrust::device_vector<float> d_logLuminance;
    thrust::device_vector<unsigned int> d_cdf;
    float min_logLum, max_logLum;

    preProcess(imgData, imgHDR, d_logLuminance, d_x, d_y, d_cdf, numRows, numCols, input_file);

    // cuda_histogram_and_prefixsum(thrust::raw_pointer_cast(d_logLuminance.data()),
    //                              thrust::raw_pointer_cast(d_cdf.data()),
    //                              min_logLum, max_logLum, numRows, numCols);
    cuda_histogram_and_prefixsum(d_logLuminance, d_cdf,
                                 min_logLum, max_logLum, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    postProcess(output_file, imgHDR, d_logLuminance, d_x, d_y, d_cdf,
                numRows, numCols, min_logLum, max_logLum);
}