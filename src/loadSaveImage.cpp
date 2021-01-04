#include <iostream>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void loadImageRGBA(const std::string &filename,
                   thrust::device_vector<uchar4> &imageVector,
                   size_t &numRows, size_t &numCols)
{
    cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    if (image.channels() != 3)
    {
        std::cerr << "Image must be color!" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (!image.isContinuous())
    {
        std::cerr << "Image isn't continuous!" << std::endl;
        exit(EXIT_FAILURE);
    }
    numRows = image.rows;
    numCols = image.cols;

    cv::Mat imageRGBA;
    cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

    uchar4 *cvPtr = imageRGBA.ptr<uchar4>(0);
    imageVector.assign(cvPtr, cvPtr + numRows * numCols); //deep copy of cv::Mat image data
}

void loadImageHDR(const std::string &filename,
                  thrust::device_vector<float> &imageVector,
                  size_t &numRows, size_t &numCols)
{
    cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    if (image.type() != CV_32FC3)
        image.convertTo(image, CV_32FC3);

    if (image.empty())
    {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }
    if (image.channels() != 3)
    {
        std::cerr << "Image must be color!" << std::endl;
        exit(1);
    }
    if (!image.isContinuous())
    {
        std::cerr << "Image isn't continuous!" << std::endl;
        exit(1);
    }
    numRows = image.rows;
    numCols = image.cols;

    float *cvPtr = image.ptr<float>(0);
    imageVector.assign(cvPtr, cvPtr + numRows * numCols * 3); //deep copy of cv::Mat image data
}

void saveImageRGBA(const thrust::device_vector<uchar4> &imageVector,
                   const size_t numRows, const size_t numCols,
                   const std::string &output_file)
{
    thrust::host_vector<uchar4> h_image = imageVector;
    cv::Mat output(numRows, numCols, CV_8UC4, (void *)h_image.data());
    cv::cvtColor(output, output, cv::COLOR_RGBA2BGR);
    cv::imwrite(output_file.c_str(), output);
}

void saveImageGrey(const thrust::device_vector<unsigned char> &imageVector,
                   const size_t numRows, const size_t numCols,
                   const std::string &output_file)
{
    thrust::host_vector<unsigned char> h_image = imageVector;
    cv::Mat output(numRows, numCols, CV_8UC1, (void *)h_image.data());
    cv::imwrite(output_file.c_str(), output);
}

void saveImageHDR(const thrust::device_vector<float> &imageVector,
                  const size_t numRows, const size_t numCols,
                  const std::string &output_file)
{
    thrust::host_vector<float> h_image = imageVector;
    cv::Mat imageHDR(numRows, numCols, CV_32FC3, (void *)h_image.data());
    cv::imwrite(output_file.c_str(), imageHDR * 255);
}
