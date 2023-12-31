cmake_minimum_required(VERSION 3.13)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

project(OneAPI_Hackathon LANGUAGES C CXX)

set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpx)

find_package(MKL CONFIG REQUIRED)

find_path(SYCL_HEADERS CL/sycl.hpp PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux/include/sycl PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_path(DPCPP_HEADERS oneapi/dpl/iterator PATHS $ENV{ONEAPI_ROOT}/dpl/latest/linux/include PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_library(SYCL_LIB sycl PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux PATH_SUFFIXES lib NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)

find_package(OpenCLHeaders REQUIRED)

# 生成一个名为 fft 的可执行文件，该文件由 main.cpp 编译而来
add_executable(fft main.cpp)

target_compile_options(fft PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(fft PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(fft PUBLIC $<LINK_ONLY:MKL::MKL>)

# link intel SYCL
target_compile_options(fft PRIVATE -fsycl)
target_include_directories(fft SYSTEM PRIVATE
  ${SYCL_HEADERS}
  ${DPCPP_HEADERS}
  ${InferenceEngine_INCLUDE_DIRS}
)
target_link_libraries(fft PUBLIC
  OpenCL::Headers dl mkl_sycl ${SYCL_LIB} fftw3
)