#ifndef LOADSAVEIMAGE_H__
#define LOADSAVEIMAGE_H__

void loadImageHDR(const std::string &filename,
                  float **imagePtr,
                  size_t *numRows, size_t *numCols);

void loadImageRGBA(const std::string &filename,
                   uchar4 **imagePtr,
                   size_t *numRows, size_t *numCols);

void loadImageGrey(const std::string &filename,
                   unsigned char **imagePtr,
                   size_t *numRows, size_t *numCols);

void saveImageRGBA(const uchar4 *const image,
                   const int numRows, const int numCols,
                   const std::string &output_file);

void saveImageGrey(const unsigned char *image,
                   const int numRows, const int numCols,
                   const std::string &output_file);

void saveImageHDR(const float *const image,
                  const int numRows, const int numCols,
                  const std::string &output_file);

#endif
