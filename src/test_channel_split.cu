#include <opencv2/imgcodecs.hpp>
#include <thrust/device_vector.h>
#include <nppi_data_exchange_and_initialization.h>

void test_channel_split(const std::string &input_file, const std::string &output_file)
{
    size_t cols, rows;
    thrust::device_vector<unsigned char> imgInput;
    thrust::device_vector<unsigned char> imgOutput;

    // read input image in BGR format
    cv::Mat image = cv::imread(input_file.c_str(), cv::IMREAD_COLOR);
    rows = image.rows;
    cols = image.cols;
    unsigned char *cvPtr = image.ptr<unsigned char>(0);
    imgInput.assign(cvPtr, cvPtr + 3 * rows * cols);

    // allocate memory for output image and individualchannels
    size_t numPixels = rows * cols;
    imgOutput.resize(numPixels * 3);
    thrust::device_vector<unsigned char> d_red(numPixels), d_green(numPixels), d_blue(numPixels);

    // split input image into individual channels
    const unsigned char *pSrc = thrust::raw_pointer_cast(imgInput.data());
    int nSrcStep = 3 * cols * sizeof(unsigned char);
    unsigned char *const aDst[3] = {thrust::raw_pointer_cast(d_blue.data()),
                                    thrust::raw_pointer_cast(d_green.data()),
                                    thrust::raw_pointer_cast(d_red.data())};
    int nDstStep = cols * sizeof(unsigned char);
    NppiSize oSizeROI{.width = cols, .height = rows};
    nppiCopy_8u_C3P3R(pSrc, nSrcStep, aDst, nDstStep, oSizeROI);

    // combine individual channels into output image
    const unsigned char *const aSrc[3] = {thrust::raw_pointer_cast(d_blue.data()),
                                          thrust::raw_pointer_cast(d_green.data()),
                                          thrust::raw_pointer_cast(d_red.data())};
    nSrcStep = cols * sizeof(unsigned char);
    unsigned char *pDst = thrust::raw_pointer_cast(imgOutput.data());
    nDstStep = 3 * cols * sizeof(unsigned char);
    nppiCopy_8u_P3C3R(aSrc, nSrcStep, pDst, nDstStep, oSizeROI);

    // write output image in BGR format
    thrust::host_vector<unsigned char> h_image = imgOutput;
    cv::Mat output(rows, cols, CV_8UC3, (void *)h_image.data());
    cv::imwrite(output_file.c_str(), output);
}