#include <iostream>
#include <cassert>

#include <cuda_runtime.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include "../include/utils.h"
#include "../include/loadSaveImage.h"

void preProcess(uchar4 **sourceImg, uchar4 **destImg, uchar4 **blendedImg,
                size_t &numRowsSource, size_t &numColsSource,
                size_t &numRowsDest, size_t &numColsDest,
                const std::string &source_filename,
                const std::string &dest_filename)
{

    //make sure the context initializes ok
    checkCudaErrors(cudaFree(0));

    loadImageRGBA(source_filename, sourceImg, &numRowsSource, &numColsSource);
    loadImageRGBA(dest_filename, destImg, &numRowsDest, &numColsDest);
    *blendedImg = new uchar4[numRowsDest * numColsDest];

    assert(numRowsSource == numRowsDest);
    assert(numColsSource == numColsDest);
}

void opencv_hack(const uchar4 *const sourceImg,
                 const uchar4 *const destImg,
                 uchar4 *const blendedImg,
                 const int numRowsSource, const int numColsSource,
                 const int numRowsDest, const int numColsDest,
                 int flag)
{
    // load in source and destination images
    cv::Mat source(numRowsSource, numColsSource, CV_8UC4, (void *)sourceImg);
    cv::cvtColor(source, source, cv::COLOR_RGBA2BGR);

    cv::Mat dest(numRowsDest, numColsDest, CV_8UC4, (void *)destImg);
    cv::cvtColor(dest, dest, cv::COLOR_RGBA2BGR);

    // create mask (assuming source image is the same size as destination image
    // and is white outisde the region of interest)
    cv::Mat mask;
    cv::cvtColor(source, mask, cv::COLOR_BGR2GRAY);
    mask.setTo(0, mask < 255);
    mask = ~mask;

    // // calculate centroid of source image (i.e. where to place it in the destination image)
    cv::Moments m = cv::moments(mask);
    cv::Point p;
    p.x = m.m10 / m.m00;
    p.y = m.m01 / m.m00;

    // apply seamless clone
    cv::Mat blended;
    cv::seamlessClone(source, dest, mask, p, blended, flag);
    cv::cvtColor(blended, blended, cv::COLOR_BGR2RGBA);

    unsigned char *cvPtr = blended.ptr<unsigned char>(0);
    for (int i = 0; i < numRowsDest * numColsDest; ++i)
    {
        blendedImg[i].x = cvPtr[4 * i + 0];
        blendedImg[i].y = cvPtr[4 * i + 1];
        blendedImg[i].z = cvPtr[4 * i + 2];
        blendedImg[i].w = cvPtr[4 * i + 3];
    }
}

void postProcess(const std::string &output_file, const uchar4 *const blendedImg,
                 const int numRowsDest, const int numColsDest)
{
    saveImageRGBA(blendedImg, numRowsDest, numColsDest, output_file);
}

void seamless_clone(const std::string &input_file, const std::string &dest_file, const std::string &output_file)
{
    size_t numRowsSource, numColsSource;
    size_t numRowsDest, numColsDest;
    uchar4 *h_sourceImg, *h_destImg, *h_blendedImg;

    preProcess(&h_sourceImg, &h_destImg, &h_blendedImg,
               numRowsSource, numColsSource, numRowsDest, numColsDest,
               input_file, dest_file);

    opencv_hack(h_sourceImg, h_destImg, h_blendedImg,
                numRowsSource, numColsSource, numRowsDest, numColsDest, 1);

    postProcess(output_file, h_blendedImg, numRowsDest, numColsDest);

    delete[] h_sourceImg;
    delete[] h_blendedImg;
    delete[] h_destImg;
}
