# 基于 OneMKL 库的高性能二维傅里叶变换实践及性能分析

> 本样例代码位于 [github 仓库](https://github.com/truc0/oneapi-hackathon-2023/tree/main/demo)

本样例基于 OneMKL 库实现了指定大小的二维随机矩阵生成，通过 OneMKL 库提供的快速傅里叶变换接口实现了高性能 real to complex （实数输入、复数输出）的单精度二维傅里叶变换，同时提供了基于 FFTW3 库的等价实现并进行了性能对比。

读者可以通过本样例了解 OneMKL 库在 C++ 程序中的基本用法，了解 OneMKL 的性能优势。

## OneMKL 库简介及安装

OneMKL 是 Intel OneAPI 的数学计算库，该库对常见的数学运算进行了高度优化和并行化，能够在 CPU 上实现高速运算，并且包含 SYCL 接口为 CPU/GPU 异构平台计算提供支持。OneMKL 库包含对稠密矩阵代数、稀疏矩阵代数、随机数生成、快速傅里叶变换等常用数学计算的高性能实现，能为计算密集型程序提供强大、易用的数学计算接口。

使用 OneMKL 库需要首先安装 Intel OneAPI，本样例中的代码在 Intel OneAPI 2023.2 环境下进行开发。OneMKL 库的下载和安装方式可以在 [Intel OneMKL 官方网站](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) 上找到。OneMKL 库同时是 [Intel OneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) 中的一部分，本样例中会使用 Intel OneAPI Toolkit 中的其他套件，**推荐直接[安装 Intel OneAPI Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)** 以获得更好的开发体验。

## 编写第一个 OneMKL 程序

本样例使用 CMake 进行编译、链接配置，请确保已安装 3.13 版本以上的 CMake，本样例已在 Cmake 3.20 上测试。本样例的 [CMakeLists.txt](https://github.com/truc0/oneapi-hackathon-2023/tree/main/demo/CMakeLists.txt) 主要参考 [Intel CMake Config for OneMKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/cmake-config-for-onemkl.html) 中的设置，详细配置可参考 [OneMKL 开发者文档](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/overview.html)。

### CMake 配置

CMake 配置文件是位于项目根目录的 `CMakeLists.txt` 文件，该文件对编译选项、链接选项、编译产物等进行了详细配置。如果你只想要了解 OneMKL 库的用法，你可以跳过此段，直接复制样例中的 CMakeLists.txt 修改使用。

首先，我们需要在 `CMakeLists.txt` 中声明该项目支持的 CMake 最低版本等基础信息，具体如下：

```cmake
cmake_minimum_required(VERSION 3.13)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

project(OneAPI_Hackathon LANGUAGES C CXX)

set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpx)
```

其中 `icc` 和 `icpx` 为 Intel OneAPI Base Toolkit 提供的编译器。

然后，我们需要声明本样例中可能用到的接口，主要包括 MKL、DPCPP、SYCL 的头文件：

```cmake
find_package(MKL CONFIG REQUIRED)

find_path(SYCL_HEADERS CL/sycl.hpp PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux/include/sycl PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_path(DPCPP_HEADERS oneapi/dpl/iterator PATHS $ENV{ONEAPI_ROOT}/dpl/latest/linux/include PATH_SUFFIXES / NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)
find_library(SYCL_LIB sycl PATHS $ENV{ONEAPI_ROOT}/compiler/latest/linux PATH_SUFFIXES lib NO_CMAKE_FIND_ROOT_PATH NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH)

find_package(OpenCLHeaders REQUIRED)
```

接下来，我们需要声明本样例的编译产物：

```cmake
# 生成一个名为 fft 的可执行文件，该文件由 main.cpp 编译而来
add_executable(fft main.cpp)
```

最后，声明我们需要链接的库：

```cmake
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
```

### 试试编译运行

在项目根目录下新建 `main.cpp` 文件，写入以下内容：

```cpp
#include <iostream>

int main()
{
    std::cout << "Hello world!" << std::endl;
    return 0;
}
```

在项目根目录运行以下命令：

```bash
# 新建并切换到 build 目录
mkdir -p build && cd build
# 编译
cmake .. && cmake --build .
# 运行
./fft
```

如果 Intel OneAPI Base Toolkit 安装无误且 CMake 相关配置正确，项目应该能够正常编译运行并输出 `Hello world!`。

### 程序结构

本样例采用了常见的 C++ 目录格式，头文件（`.h`、`.hpp`）位于 `includes` 目录下，源码文件（`.c`、`.cpp`）位于 `src` 目录下。程序的入口位于 `main.cpp`，`main.cpp` 放置在项目根目录下。

### OneMKL 初探：生成随机数

编辑 `main.cpp`：

```cpp
#include <iostream>
#include <vector>
#include "CL/sycl.hpp"
#include "oneapi/mkl/rng.hpp"

const auto size = 5;

int main()
{
    // 初始化 SYCL，使用默认选择器选择设备
    sycl::queue queue;
    // 初始化分配器，设置为分配浮点数且分配的内存为设备共享内存
    // 当 queue 选择的设备为 GPU 或 FPGA 时，OneAPI 的 SYCL 接口
    // 会自动在 GPU 和主机节点（Host）中同步设备共享内存
    sycl::usm_allocator<float, sycl::usm::alloc::shared> allocator(queue.get_context(), queue.get_device());
    // 初始化一个 vector 用于存放结果
    std::vector<float, decltype(allocator)> r(size, allocator);

    // 使用默认的生成引擎
    oneapi::mkl::rng::default_engine engine(queue);
    // 设置为平均分布
    oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate> distr;

    // 异步生成随机数
    auto event = oneapi::mkl::rng::generate(distr, engine, size, r.data());
    // 等待生成完毕
    event.wait();

    // 输出
    for (const auto num: r)
    {
        std::cout << num << std::endl;
    }

    return 0;
}
```

重新编译运行：

```bash
# 请确保你在 build 目录下
cmake .. 
cmake --build .
./fft
```

若程序正常则会输出 5 个随机浮点数。

## 二维傅里叶变换

本样例需要实现 real to complex （实数域到复数域）的单精度二维傅里叶变换，为实现方便，输入和输出均以一维矩阵的形式存储。本样例中的输入是 `N * N` 的矩阵，取 N 为 2048。

### 二维随机矩阵生成

将上一节中的代码包装为函数：

```cpp
const auto N = 2048;
const auto SIZE = N * N;

using RealAllocatorType = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
using RealVectorType = std::vector<float, RealAllocatorType>;

RealVectorType generate_random_numbers(sycl::queue queue, unsigned size)
{
    // 初始化分配器，设置为分配浮点数且分配的内存为设备共享内存
    RealAllocatorType allocator(queue.get_context(), queue.get_device());
    // 初始化一个 vector 用于存放结果
    RealVectorType r(size, allocator);

    // 使用默认的生成引擎
    oneapi::mkl::rng::default_engine engine(queue);
    // 设置为平均分布
    oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate> distr;

    // 异步生成随机数
    auto event = oneapi::mkl::rng::generate(distr, engine, size, r.data());
    // 等待生成完毕
    event.wait();

    return r;
}
```

### OneMKL 运行傅里叶变换

OneMKL 库中提供了多维傅里叶变换函数，并提供了丰富的配置选项。使用 OneMKL 库的傅里叶变换接口需要首先使用 `descriptor` 的方式进行配置，在本样例中，傅里叶变换的主要参数如下：

- 二维矩阵，N*N
- 单精度
- 实数域到复数域
- 傅里叶正向变换
- 将变换后结果存储到新数组中

在函数中对应的配置代码如下：

```cpp
// 创建一个描述符（descriptor），配置为单精度，实数域到复数域
// 配置为 N * N 的傅里叶变换
oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                oneapi::mkl::dft::domain::REAL>
    desc({N, N});
// 
desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, static_cast<std::int64_t>(1));
// 由于实数域到复数域的傅里叶变换具有共轭性，这里需要指定
desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
// 配置为不原地存储，将结果存储到新数组中
desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
// 提交配置
desc.commit(queue);
```

配置完成后，执行傅里叶变换：

```cpp
using ComplexAllocatorType = sycl::usm_allocator<std::complex<float>, sycl::usm::alloc::shared>;
using ComplexVectorType = std::vector<std::complex<float>, ComplexAllocatorType>;

// 由于配置了非原地存储，我们需要新建一个数组
ComplexVectorType dst(N * N, allocator);

// compute_forward，前向傅里叶变换，即正变换
// decltype(desc) 为推断 descr 的类型
// input.data() 和 dst.data() 为取出 std::vector 中存储数据的地址
oneapi::mkl::dft::compute_forward<
    decltype(desc), float, std::complex<float>>(desc, input.data(), dst.data())
    .wait();
```

至此，我们使用 OneMKL 库完成了二维快速傅里叶变换，完整的代码文件参见 [github 仓库](https://github.com/truc0/oneapi-hackathon-2023/tree/main/demo/main.cpp)。

注意由于实数序列的频域具有共轭性，在输出中只有前 $\frac{N}{2} + 1$ 列数据有效。

## 性能对比及正确性验证

我们采用 FFTW3 库作为性能对比和正确性验证的快速傅里叶变换参考实现，这里给出对应的代码，完整的代码文件参见 [github 仓库](https://github.com/truc0/oneapi-hackathon-2023/tree/main/demo)。：

```cpp
ComplexVectorType use_fftw(const unsigned N, RealVectorType &input, sycl::queue &queue)
{
    const auto size = N * N;
    const auto validCol = N / 2 + 1;
    auto allocator = ComplexAllocatorType(queue.get_context(), queue.get_device());

    // 为输入和输出分配空间
    auto in = fftwf_alloc_real(size);
    auto out = fftwf_alloc_complex(size);

    // 复制输入
    for (unsigned i = 0; i < input.size(); ++i)
    {
        in[i] = input[i];
    }

    // 设置参数：二维，N*N，实数域到复数域
    auto plan = fftwf_plan_dft_r2c_2d(N, N, in, out, FFTW_ESTIMATE);

    // 运行 fft
    fftwf_execute(plan);

    // fftw 输出格式与 OneMKL 不同，这里进行格式统一
    // fftw 只有前 (N/2 + 1) * rows 个数据有效
    ComplexVectorType output(size, allocator);
    for (unsigned row = 0; row < N; ++row)
    {
        for (unsigned col = 0; col < validCol; ++col)
        {
            output[row * N + col] = {out[row * validCol + col][0], out[row * validCol + col][1]};
        }
    }

    fftwf_free(in);
    fftwf_free(out);
    return output;
}
```

对比代码：

```cpp
sycl::queue queue;
auto input = generate_random_numbers(queue, SIZE);
auto allocator = ComplexAllocatorType(queue.get_context(), queue.get_device());

// variables
ComplexVectorType oneResult(size, allocator);
std::chrono::nanoseconds oneTime;
ComplexVectorType fftwResult(size, allocator);
std::chrono::nanoseconds fftwTime;

// execute
{
    auto t1 = std::chrono::high_resolution_clock::now();
    oneResult = use_onemkl(N, input, queue);
    auto t2 = std::chrono::high_resolution_clock::now();
    oneTime = t2 - t1;
}
{
    auto t1 = std::chrono::high_resolution_clock::now();
    fftwResult = use_fftw(N, input, queue);
    auto t2 = std::chrono::high_resolution_clock::now();
    fftwTime = t2 - t1;
}

// compare
long double diff = 0;
const validCol = N / 2 + 1;
for (unsigned row = 0; row < N; ++row)
{
    for (unsigned col = 0; col < validCol; ++col)
    {
        diff += std::abs(oneResult[row * N + col] - fftwResult[row * N + col]);
    }
}

// output
std::cout << "OneMKLFFTExecutor time: " << oneTime.count() << " ns" << std::endl;
std::cout << "FFTWFFTExecutor time: " << fftwTime.count() << " ns" << std::endl;
std::cout << "Difference of oneMKL and fftw: " << diffOne << std::endl;
```

运行后输出的格式类似：

```txt
OneMKLFFTExecutor time: 25526224 ns
FFTWFFTExecutor time: 36094255 ns
Difference of oneMKL and fftw: 0
```
