#include <iostream>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

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

void preProcess(float **imgPtr, float **imgPtrHDR,
                float **d_logLuminance, float **d_x, float **d_y,
                unsigned int **d_cdf,
                size_t &rows, size_t &cols,
                const std::string &filename)
{
    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    // allocate and load input image
    loadImageHDR(filename, imgPtr, &rows, &cols);
    // allocate output image
    *imgPtrHDR = new float[rows * cols * 3];

    //first thing to do is split incoming BGR float data into separate channels
    size_t numPixels = rows * cols;
    float *red = new float[numPixels];
    float *green = new float[numPixels];
    float *blue = new float[numPixels];

    //loadImageHDR keeps BGR format
    for (size_t i = 0; i < numPixels; ++i)
    {
        blue[i] = (*imgPtr)[3 * i];
        green[i] = (*imgPtr)[3 * i + 1];
        red[i] = (*imgPtr)[3 * i + 2];
    }

    float *d_red, *d_green, *d_blue; //RGB space

    size_t channelSize = sizeof(float) * numPixels;
    checkCudaErrors(cudaMalloc(&d_red, channelSize));
    checkCudaErrors(cudaMalloc(&d_green, channelSize));
    checkCudaErrors(cudaMalloc(&d_blue, channelSize));
    checkCudaErrors(cudaMalloc(d_x, channelSize));
    checkCudaErrors(cudaMalloc(d_y, channelSize));
    checkCudaErrors(cudaMalloc(d_logLuminance, channelSize));

    checkCudaErrors(cudaMemcpy(d_red, red, channelSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_green, green, channelSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_blue, blue, channelSize, cudaMemcpyHostToDevice));

    //convert from RGB space to chrominance/luminance space xyY
    dim3 blockSize(32, 16, 1);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y, 1);
    rgb_to_xyY_kernel<<<gridSize, blockSize>>>(d_red, d_green, d_blue,
                                               *d_x, *d_y, *d_logLuminance,
                                               .0001f, rows, cols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    //allocate memory for the cdf of the histogram
    checkCudaErrors(cudaMalloc(d_cdf, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMemset(*d_cdf, 0, sizeof(unsigned int) * numBins));

    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
    delete[] red;
    delete[] green;
    delete[] blue;
}

void postProcess(const std::string &output_file, float *const imageHDR,
                 const float *const d_logLuminance,
                 const float *const d_x, const float *const d_y,
                 const unsigned int *const d_cdf,
                 const int numRows, const int numCols,
                 const float min_log_Y, const float max_log_Y)
{
    //first normalize the cdf to a maximum value of 1
    //this is how we compress the range of the luminance channel
    float *d_cdf_normalized;
    checkCudaErrors(cudaMalloc(&d_cdf_normalized, sizeof(float) * numBins));

    int numThreads = 192;
    int numBlocks = (numBins + numThreads - 1) / numThreads;
    normalize_cdf<<<numBlocks, numThreads>>>(d_cdf, d_cdf_normalized, numBins);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    //allocate memory for the output RGB channels
    size_t numPixels = numRows * numCols;

    float *h_red, *h_green, *h_blue;
    float *d_red, *d_green, *d_blue;

    h_red = new float[numPixels];
    h_green = new float[numPixels];
    h_blue = new float[numPixels];

    checkCudaErrors(cudaMalloc(&d_red, sizeof(float) * numPixels));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(float) * numPixels));
    checkCudaErrors(cudaMalloc(&d_blue, sizeof(float) * numPixels));

    float log_Y_range = max_log_Y - min_log_Y;

    dim3 blockSize(32, 16, 1);
    dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
                  (numRows + blockSize.y - 1) / blockSize.y);
    //map each luminance value to its new value and then transform back to RGB space
    tonemap_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_logLuminance,
                                            d_cdf_normalized,
                                            d_red, d_green, d_blue,
                                            min_log_Y, max_log_Y,
                                            log_Y_range, numBins,
                                            numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_red, d_red, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_green, d_green, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_blue, d_blue, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));

    //recombine the image channels
    for (size_t i = 0; i < numPixels; ++i)
    {
        imageHDR[3 * i + 0] = h_blue[i];
        imageHDR[3 * i + 1] = h_green[i];
        imageHDR[3 * i + 2] = h_red[i];
    }
    saveImageHDR(imageHDR, numRows, numCols, output_file);

    //cleanup
    checkCudaErrors(cudaFree(d_cdf_normalized));
    delete[] h_red;
    delete[] h_green;
    delete[] h_blue;
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

void cuda_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf,
                                  float &min_logLum, float &max_logLum,
                                  const int numRows, const int numCols)
{
    int numPixels = numRows * numCols;

    // 1) find the minimum and maximum value in the input logLuminance channel
    //    store in min_logLum and max_logLum
    thrust::device_ptr<const float> dev_ptr_logLuminance(d_logLuminance);
    max_logLum = thrust::reduce(thrust::device, dev_ptr_logLuminance, dev_ptr_logLuminance + (size_t)numPixels, max_logLum, thrust::maximum<float>());
    min_logLum = thrust::reduce(thrust::device, dev_ptr_logLuminance, dev_ptr_logLuminance + (size_t)numPixels, min_logLum, thrust::minimum<float>());

    // 2) subtract them to find the range
    float logLum_range = max_logLum - min_logLum;

    // 3) generate a histogram of all the values in the logLuminance channel using
    //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    int NumThreadsPerBlock = 512;
    int NumBlocks = (numPixels + NumThreadsPerBlock - 1) / NumThreadsPerBlock;
    dim3 blockSize(NumThreadsPerBlock, 1, 1);
    dim3 gridSize(NumBlocks, 1, 1);
    simple_histo_kernel<<<gridSize, blockSize>>>(d_logLuminance, d_cdf, min_logLum, logLum_range, numBins, numPixels);

    // 4) Perform an exclusive scan (prefix sum) on the histogram to get
    //    the cumulative distribution of luminance values
    thrust::device_ptr<unsigned int> dev_ptr_cdf(d_cdf);
    thrust::exclusive_scan(thrust::device, dev_ptr_cdf, dev_ptr_cdf + numBins, dev_ptr_cdf);
}

void tonemap_HDR(const std::string &input_file, const std::string &output_file)
{
    size_t numRows, numCols;

    float *imgData, *imgHDR;
    float *d_x, *d_y;
    float *d_logLuminance;
    unsigned int *d_cdf;
    float min_logLum, max_logLum;

    preProcess(&imgData, &imgHDR, &d_logLuminance, &d_x, &d_y, &d_cdf, numRows, numCols, input_file);

    cuda_histogram_and_prefixsum(d_logLuminance, d_cdf, min_logLum, max_logLum, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    postProcess(output_file, imgHDR, d_logLuminance, d_x, d_y, d_cdf,
                numRows, numCols, min_logLum, max_logLum);

    checkCudaErrors(cudaFree(d_logLuminance));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_cdf));
    delete[] imgData;
    delete[] imgHDR;
}