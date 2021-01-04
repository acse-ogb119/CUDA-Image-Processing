#include <iostream>

#include <cuda_runtime.h>
#include <nppi_color_conversion.h>

#include "../include/utils.h"
#include "../include/loadSaveImage.h"

void preProcess(uchar4 **h_rgbaImage, unsigned char **h_greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                size_t &rows, size_t &cols,
                const std::string &filename)
{
    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    // allocate and load input image
    loadImageRGBA(filename, h_rgbaImage, &rows, &cols);
    // allocate output image
    *h_greyImage = new unsigned char[rows * cols];

    //allocate memory on the device for both input and output
    size_t numPixels = rows * cols;
    checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));

    //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(*d_rgbaImage, *h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
    //set output array on GPU to all zeros
    checkCudaErrors(cudaMemset(*d_greyImage, 0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string &output_file, unsigned char *const h_greyImage, const unsigned char *const d_greyImage,
                 const int rows, const int cols)
{
    size_t numPixels = rows * cols;
    checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    saveImageGrey(h_greyImage, rows, cols, output_file);
}

__global__ void rgba_to_grayscale_kernel(const uchar4 *const rgbaImage,
                                         unsigned char *const greyImage,
                                         const int count)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < count)
    {
        uchar4 rgbaPixel = rgbaImage[tid];
        float ChannelSum = .299f * rgbaPixel.x + .587f * rgbaPixel.y + .114f * rgbaPixel.z;
        greyImage[tid] = ChannelSum;
    }
}

void cuda_rgba_to_grayscale(const uchar4 *const d_rgbaImage, unsigned char *const d_greyImage,
                            int numRows, int numCols)
{
    int blockSize, minGridSize, gridSize;
    int count = numCols * numRows;

    // not necessarily *optimal* block & grid sizes... but will be pretty close for a simple kernel like this
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void *)rgba_to_grayscale_kernel,
        0,
        count);
    gridSize = (count + blockSize - 1) / blockSize;

    rgba_to_grayscale_kernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, count);
}

void nppi_rgba_to_grayscale(const unsigned char *const d_rgbaImage, unsigned char *const d_greyImage,
                            int numRows, int numCols)
{
    NppiSize size;
    size.width = numCols;
    size.height = numRows;
    NppStatus status = nppiRGBToGray_8u_AC4C1R(d_rgbaImage, numCols * 4, d_greyImage, numCols, size);
}

void convert_grayscale(const std::string &input_file, const std::string &output_file)
{
    size_t numRows, numCols;

    uchar4 *h_rgbaImage, *d_rgbaImage;
    unsigned char *h_greyImage, *d_greyImage;

    preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, numRows, numCols, input_file);

    cuda_rgba_to_grayscale(d_rgbaImage, d_greyImage, numRows, numCols);
    // nppi_rgba_to_grayscale((unsigned char *)d_rgbaImage, d_greyImage, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    postProcess(output_file, h_greyImage, d_greyImage, numRows, numCols);

    //cleanup
    checkCudaErrors(cudaFree(d_rgbaImage));
    checkCudaErrors(cudaFree(d_greyImage));
    delete[] h_rgbaImage;
    delete[] h_greyImage;
}