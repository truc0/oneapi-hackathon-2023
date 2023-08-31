# oneapi-hackathon-2023

> Intel Oneapi Hackathon 2023 - optimization of single percision floating point FFT

## 运行

本程序依赖于 OneMKL 和 FFTW 库，请确保已经安装。OneMKL 库安装参见[官方下载页](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)，FFTW 库的安装可以参考[官方文档](https://www.fftw.org/fftw3_doc/Installation-and-Customization.html)从源码安装。

本程序同时需要 3.13 以上版本的 CMake 支持。

```bash
# 如果是 Debian/Ubuntu 系统，可以考虑使用 apt 安装
sudo apt install libfftw3-dev
```

### 编译运行

```bash
# 编译
mkdir -p build && cd build
cmake .. && cmake --build .
# 运行
./fft #运行程序
```

## 程序说明

本程序使用 OneMKL 库、FFTW 库和手写二重循环三种方式实现了二维 real to complex 傅里叶变换，并对他们的性能进行了分析。

程序提供了 `Generator` 和 `Executor` 抽象，`Generator` 使用 OneMKL 库中提供的 RNG (Random Number Generation) 模块进行随机数生成，在本程序中生成平均分布的 2048 * 2048 个随机浮点数。`Executor` 则是对二维 real to complex 实现的抽象，提供了 `execute` 接口，用于执行傅里叶变换。

本程序中有三种实现方式，Executor 名称和依赖关系如下：

- OneExecutor - OneMKL 库实现
- FFTWExecutor - FFTW 库实现
- SimpleExecutor - 无优化实现

在主程序中提供了 `run_once` 和 `run_multiple` 两个函数，分别用于单次运行和多次运行 Executor 并输出结果分析。

## 性能分析

### 单次运行

> 单次运行需要注释 main 函数中的 run_multiple 调用，并确保 run_once 没有被注释

SimpleExecutor 运行时间较长，因此默认不会运行，如果需要运行可以通过以下命令编译：

```bash
# 编译
mkdir -p build && cd build
cmake -DENABLE_SIMPLE=ON .. && cmake --build .
# 运行
./fft #运行程序
```

单次运行性能如下:

```txt
OneMKLFFTExecutor time: 15891658 ns
FFTWFFTExecutor time: 30341569 ns
SimpleFFTExecutor time: 71197696456 ns
Difference of oneMKL and fftw: 0
Difference of simple and fftw: 4878.95
Difference of oneMKL and simple: 4878.95
```

由于测试用的机器没有 GPU，OneMKL 和 FFTW 都运行在 CPU 上，可以明显看出 OneMKL 库的实现最快，SimpleExecutor 的实现最慢。

### 多次运行

> 多次运行需要注释 main 函数中的 run_once 调用，并确保 run_multiple 没有被注释

SimpleExecutor 运行时间教程，因此在多次运行时不采用此方法。

多次运行性能如下：

```txt
1024/1024 finished.
OneMKLFFTExecutor time
===============================
Avg:    29314686 ns
Max:    97382975 ns
Min:    15278571 ns

FFTWFFTExecutor time
===============================
Avg:    46139556 ns
Max:    65707207 ns
Min:    30531812 ns

Difference
===============================
Avg difference: 0
Max difference: 0
Min difference: 0
```

### 运行配置

- Ubuntu 22.04.3 LTS x86_64
- Kernel: 5.15.0-78-generic
- Intel i9-9880H (16) @ 4.800GHz
- Intel OneAPI 2023.2
