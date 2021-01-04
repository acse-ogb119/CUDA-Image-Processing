#include <iostream>

#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <thrust/device_vector.h>

#include "../include/utils.h"
#include "../include/loadSaveImage.h"

void preProcess(thrust::device_vector<uchar4> &rgbaImage,
                thrust::device_vector<unsigned char> &greyImage,
                size_t &rows, size_t &cols,
                const std::string &filename)
{
    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    // allocate and load input image
    loadImageRGBA(filename, rgbaImage, rows, cols);
    // allocate output image
    greyImage.resize(rows * cols);
}

void postProcess(const std::string &output_file,
                 thrust::device_vector<unsigned char> &greyImage,
                 const size_t rows, const size_t cols)
{
    saveImageGrey(greyImage, rows, cols, output_file);
}

__global__ void rgba_to_grayscale_kernel(const uchar4 *const rgbaImage,
                                         unsigned char *const greyImage,
                                         const int count)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < count)
    {
        // int ptr = 4 * tid;
        // float ChannelSum = .299f * rgbaImage[ptr] + .587f * rgbaImage[ptr + 1] + .114f * rgbaImage[ptr + 2];
        uchar4 rgbaPixel = rgbaImage[tid];
        float ChannelSum = .299f * rgbaPixel.x + .587f * rgbaPixel.y + .114f * rgbaPixel.z;
        greyImage[tid] = ChannelSum; // FIXME: relying on implicit conversion rather than applying appropriate rounding
    }
}

void cuda_rgba_to_grayscale(const uchar4 *const d_rgbaImage,
                            unsigned char *const d_greyImage,
                            const int numRows, const int numCols)
{
    int blockSize, minGridSize, gridSize;
    int count = numCols * numRows;

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void *)rgba_to_grayscale_kernel,
        0,
        count);
    gridSize = (count + blockSize - 1) / blockSize;

    rgba_to_grayscale_kernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, count);
}

void convert_grayscale(const std::string &input_file, const std::string &output_file)
{
    size_t numRows, numCols;
    thrust::device_vector<uchar4> rgbaImage;
    thrust::device_vector<unsigned char> greyImage;

    preProcess(rgbaImage, greyImage, numRows, numCols, input_file);

    cuda_rgba_to_grayscale(thrust::raw_pointer_cast(rgbaImage.data()),
                           thrust::raw_pointer_cast(greyImage.data()),
                           numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    postProcess(output_file, greyImage, numRows, numCols);
}