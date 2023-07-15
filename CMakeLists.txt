# reference: https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-1/cmake-config-for-onemkl.html
cmake_minimum_required(VERSION 3.13)
enable_testing()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

project(OneAPI_Hackathon LANGUAGES C CXX)

set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpx)

find_package(MKL CONFIG REQUIRED)

find_path(SYCL_HEADERS CL/sycl.hpp PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux/include/sycl PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_path(DPCPP_HEADERS oneapi/dpl/iterator PATHS $ENV{ONEAPI_ROOT}/dpl/latest/linux/include PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_library(SYCL_LIB sycl PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux PATH_SUFFIXES lib NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)

if(NOT DPCPP_HEADERS)
  message(FATAL_ERROR "No DPCPP Headers for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "DPCPP Headers - ${DPCPP_HEADERS}")
endif()

if(NOT SYCL_HEADERS)
  message(FATAL_ERROR "No SYCL Headers for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "SYCL Headers - ${SYCL_HEADERS}")
endif()

if(NOT SYCL_LIB)
  message(FATAL_ERROR "No SYCL Library for oneAPI found. Please check the environment variable ONEAPI_ROOT")
else()
  message(STATUS "SYCL Library - ${SYCL_LIB}")
endif()

find_package(OpenCLHeaders REQUIRED)

# message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input

add_executable(fft main.cpp src/random.cpp)

target_compile_options(fft PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(fft PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(fft PUBLIC $<LINK_ONLY:MKL::MKL>)

# link intel SYCL
target_compile_options(fft PRIVATE -fsycl)
target_include_directories(fft SYSTEM PRIVATE
  ${SYCL_HEADERS}
  ${DPCPP_HEADERS}
  ${InferenceEngine_INCLUDE_DIRS}
  "includes/"
)
target_link_libraries(fft PUBLIC
  OpenCL::Headers
  dl
  mkl_sycl
  ${SYCL_LIB}
)

# add_test(NAME fft_test COMMAND myapp)
# if(MKL_ENV)
#   set_tests_properties(fft_test PROPERTIES ENVIRONMENT "${MKL_ENV}")
# endif()