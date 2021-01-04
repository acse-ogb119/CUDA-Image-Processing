#ifndef LOADSAVEIMAGE_H__
#define LOADSAVEIMAGE_H__

void loadImageRGBA(const std::string &filename,
                   thrust::device_vector<uchar4> &imageVector,
                   size_t &numRows, size_t &numCols);

void loadImageHDR(const std::string &filename,
                  thrust::device_vector<float> &imageVector,
                  size_t &numRows, size_t &numCols);

void saveImageRGBA(const thrust::device_vector<uchar4> &imageVector,
                   const size_t numRows, const size_t numCols,
                   const std::string &output_file);

void saveImageGrey(const thrust::device_vector<unsigned char> &imageVector,
                   const size_t numRows, const size_t numCols,
                   const std::string &output_file);

void saveImageHDR(const thrust::device_vector<float> &imageVector,
                  const size_t numRows, const size_t numCols,
                  const std::string &output_file);

#endif
