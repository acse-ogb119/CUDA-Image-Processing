SRC_DIR:=src
INCL_DIR:=include
BUILD_DIR:=build
OUT_DIR:=output

# target executable
EXE:=imgproc

# external libraries and headers
OPENCV_PATH:=/opt/opencv
OPENCV_INCLUDEPATH:=$(OPENCV_PATH)/include/opencv4
OPENCV_LIBPATH:=$(OPENCV_PATH)/lib
OPENCV_LIBS:=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_photo

CUDA_PATH:=/usr/local/cuda-11.1
CUDA_INCLUDEPATH:=$(CUDA_PATH)/include
CUDA_LIBPATH:=$(CUDA_PATH)/lib64/
CUDA_LIBS:=-lnppc -lnppicc -lnppidei

# GCC
CXX:=g++
CXXFLAGS:=-Wall -Wextra -Wpedantic
CPPFLAGS:= -MMD -MP -I$(OPENCV_INCLUDEPATH) -I$(CUDA_INCLUDEPATH)
LDFLAGS:=-L$(OPENCV_LIBPATH) -L$(CUDA_LIBPATH)
LDLIBS:=$(OPENCV_LIBS) $(CUDA_LIBS)

# NVCC
NVCC:=$(CUDA_PATH)/bin/nvcc
NVCCFLAGS:=-arch=sm_61

# Source, object and dependency files
SRC_FILES:=$(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)
OBJ_FILES:=$(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(notdir $(basename $(SRC_FILES)))))
DEP_FILES:=$(OBJ_FILES:.o=.d)

## all	: Build HW1exe program from src and include files 
.PHONY : all
all : $(EXE)

$(EXE) : $(OBJ_FILES) | output
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY : clean
clean:
	-$(RM) $(EXE) $(OBJ_FILES) $(DEP_FILES)
	rm -rf $(BUILD_DIR) $(OUT_DIR)

.PHONY : build 
build:
	mkdir -p $(BUILD_DIR)

.PHONY : output
output:
	mkdir -p $(OUT_DIR)

-include $(DEP_FILES)

# build object files
$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cu Makefile | build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@
