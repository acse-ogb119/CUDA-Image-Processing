#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../include/utils.h"
#include "../include/loadSaveImage.h"

static const int filterWidth = 9;
static const float filterSigma = 2.f;

void preProcess(thrust::device_vector<uchar4> &d_inputImageRGBA,
                thrust::device_vector<uchar4> &d_outputImageRGBA,
                thrust::device_vector<unsigned char> &d_redBlurred,
                thrust::device_vector<unsigned char> &d_red,
                thrust::device_vector<unsigned char> &d_greenBlurred,
                thrust::device_vector<unsigned char> &d_green,
                thrust::device_vector<unsigned char> &d_blueBlurred,
                thrust::device_vector<unsigned char> &d_blue,
                thrust::device_vector<float> &d_filter,
                size_t &rows, size_t &cols,
                const std::string &filename)
{
    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    // allocate and load input image
    loadImageRGBA(filename, d_inputImageRGBA, rows, cols);
    // allocate output image
    size_t numPixels = rows * cols;
    d_outputImageRGBA.assign(numPixels, make_uchar4(0, 0, 0, 0));

    //create and fill the filter we will convolve with
    thrust::host_vector<float> h_filter;
    h_filter.resize(filterWidth * filterWidth);

    float filterSum = 0.f; //for normalization
    for (int r = -filterWidth / 2; r <= filterWidth / 2; ++r)
    {
        for (int c = -filterWidth / 2; c <= filterWidth / 2; ++c)
        {
            float filterValue = expf(-(float)(c * c + r * r) / (2.f * filterSigma * filterSigma));
            h_filter[(r + filterWidth / 2) * filterWidth + c + filterWidth / 2] = filterValue;
            filterSum += filterValue; // for normalization
        }
    }
    // normalize filter
    float normalizationFactor = 1.f / filterSum;
    for (int r = -filterWidth / 2; r <= filterWidth / 2; ++r)
        for (int c = -filterWidth / 2; c <= filterWidth / 2; ++c)
            h_filter[(r + filterWidth / 2) * filterWidth + c + filterWidth / 2] *= normalizationFactor;

    //original
    d_red.resize(numPixels);
    d_green.resize(numPixels);
    d_blue.resize(numPixels);

    //blurred
    d_redBlurred.assign(numPixels, 0);
    d_greenBlurred.assign(numPixels, 0);
    d_blueBlurred.assign(numPixels, 0);

    //filter
    d_filter = h_filter;
}

void postProcess(const std::string &output_file, thrust::device_vector<uchar4> &d_outputImageRGBA,
                 const size_t rows, const size_t cols)
{
    saveImageRGBA(d_outputImageRGBA, rows, cols, output_file);
}

__global__ void gaussian_blur_kernel(const unsigned char *const inputChannel,
                                     unsigned char *const outputChannel,
                                     const int numRows, const int numCols,
                                     const float *const filter, const int filterWidth)
{
    // Set the pixel coordinate
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= numCols || i >= numRows)
        return;

    float result = 0.f;
    //For every value in the filter around the pixel (c, r)
    for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; ++filter_r)
    {
        for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; ++filter_c)
        {
            //Find the global image position for this filter position
            //clamp to boundary of the image
            int image_r = min(max(i + filter_r, 0), numRows - 1);
            int image_c = min(max(j + filter_c, 0), numCols - 1);

            float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
            float filter_value = filter[(filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2];

            result += image_value * filter_value;
        }
    }
    outputChannel[i * numCols + j] = static_cast<unsigned char>(result);
}

__global__ void separateChannels_kernel(const uchar4 *const inputImageRGBA,
                                        const int numRows, const int numCols,
                                        unsigned char *const redChannel,
                                        unsigned char *const greenChannel,
                                        unsigned char *const blueChannel)
{
    // Set the pixel coordinate
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= numCols || i >= numRows)
        return;

    int tid = i * numCols + j;

    redChannel[tid] = inputImageRGBA[tid].x;
    greenChannel[tid] = inputImageRGBA[tid].y;
    blueChannel[tid] = inputImageRGBA[tid].z;
}

__global__ void recombineChannels_kernel(const unsigned char *const redChannel,
                                         const unsigned char *const greenChannel,
                                         const unsigned char *const blueChannel,
                                         uchar4 *const outputImageRGBA,
                                         const int numRows, const int numCols)
{
    int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);

    int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //make sure we don't try and access memory outside the image
    //by having any threads mapped there return early
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    unsigned char red = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue = blueChannel[thread_1D_pos];

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] = outputPixel;
}

void cuda_gaussian_blur(const uchar4 *const d_inputImageRGBA,
                        uchar4 *const d_outputImageRGBA,
                        unsigned char *d_redBlurred, unsigned char *d_red,
                        unsigned char *d_greenBlurred, unsigned char *d_green,
                        unsigned char *d_blueBlurred, unsigned char *d_blue,
                        float *d_filter, const int filterWidth,
                        const int numRows, const int numCols)
{
    // TODO: optimizing block size and dimension
    // define the dimensions of each thread block (max = 1024 = 32*32)
    int blockW = 32;
    int blockH = 32;

    // Set reasonable block size (i.e., number of threads per block)
    dim3 blockSize(blockW, blockH);

    // Compute correct grid size (i.e., number of blocks per kernel launch)
    // from the image size and block size.
    int gridW = (numCols % blockW != 0) ? (numCols / blockW + 1) : (numCols / blockW);
    int gridH = (numRows % blockH != 0) ? (numRows / blockH + 1) : (numRows / blockH);
    dim3 gridSize(gridW, gridH);

    // Launch a kernel for separating the RGBA image into different color channels
    separateChannels_kernel<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Call your convolution kernel here 3 times, once for each color channel.
    // TODO: use streams for concurrency
    gaussian_blur_kernel<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur_kernel<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    gaussian_blur_kernel<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // Now we recombine channels results
    recombineChannels_kernel<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void gaussian_blur(const std::string &input_file, const std::string &output_file)
{
    size_t numRows, numCols;

    thrust::device_vector<uchar4> inputImageRGBA, outputImageRGBA;
    thrust::device_vector<unsigned char> red, green, blue;
    thrust::device_vector<unsigned char> redBlurred, greenBlurred, blueBlurred;
    thrust::device_vector<float> filter;

    preProcess(inputImageRGBA, outputImageRGBA, redBlurred, red, greenBlurred, green, blueBlurred, blue,
               filter, numRows, numCols, input_file);

    cuda_gaussian_blur(thrust::raw_pointer_cast(inputImageRGBA.data()), thrust::raw_pointer_cast(outputImageRGBA.data()),
                       thrust::raw_pointer_cast(redBlurred.data()), thrust::raw_pointer_cast(red.data()),
                       thrust::raw_pointer_cast(greenBlurred.data()), thrust::raw_pointer_cast(green.data()),
                       thrust::raw_pointer_cast(blueBlurred.data()), thrust::raw_pointer_cast(blue.data()),
                       thrust::raw_pointer_cast(filter.data()), filterWidth, numRows, numCols);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    postProcess(output_file, outputImageRGBA, numRows, numCols);
}