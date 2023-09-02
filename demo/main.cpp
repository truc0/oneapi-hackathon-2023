#include <iostream>
#include <vector>
#include <fftw3.h>
#include "CL/sycl.hpp"
#include "oneapi/mkl/rng.hpp"
#include "oneapi/mkl.hpp"

const auto N = 2048;
const auto SIZE = N * N;
const auto VALIDCOL = N / 2 + 1;

using RealAllocatorType = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
using RealVectorType = std::vector<float, RealAllocatorType>;
using ComplexAllocatorType = sycl::usm_allocator<std::complex<float>, sycl::usm::alloc::shared>;
using ComplexVectorType = std::vector<std::complex<float>, ComplexAllocatorType>;

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

ComplexVectorType use_onemkl(const unsigned N, RealVectorType &input, sycl::queue &queue)
{
    // 创建一个描述符（descriptor），配置为单精度，实数域到复数域
    // 配置为 N * N 的傅里叶变换
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                 oneapi::mkl::dft::domain::REAL>
        desc({N, N});
    // 配置为不原地存储，将结果存储到新数组中
    desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    // 提交配置
    desc.commit(queue);

    ComplexAllocatorType allocator(queue.get_context(), queue.get_device());
    // 由于配置了非原地存储，我们需要新建一个数组
    ComplexVectorType dst(N * N, allocator);

    // compute_forward，前向傅里叶变换，即正变换
    // decltype(desc) 为推断 descr 的类型
    // input.data() 和 dst.data() 为取出 std::vector 中存储数据的地址
    oneapi::mkl::dft::compute_forward<
        decltype(desc), float, std::complex<float>>(desc, input.data(), dst.data())
        .wait();

    return dst;
}

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
    ComplexVectorType dst(size, allocator);

    return dst;
}

int main()
{
    sycl::queue queue;
    auto input = generate_random_numbers(queue, SIZE);
    auto allocator = ComplexAllocatorType(queue.get_context(), queue.get_device());

    // variables
    ComplexVectorType oneResult(SIZE, allocator);
    std::chrono::nanoseconds oneTime;
    ComplexVectorType fftwResult(SIZE, allocator);
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
    for (unsigned row = 0; row < N; ++row)
    {
        for (unsigned col = 0; col < VALIDCOL; ++col)
        {
            diff += std::abs(oneResult[row * N + col] - fftwResult[row * N + col]);
        }
    }

    // output
    std::cout << "OneMKLFFTExecutor time: " << oneTime.count() << " ns" << std::endl;
    std::cout << "FFTWFFTExecutor time: " << fftwTime.count() << " ns" << std::endl;
    std::cout << "Difference of oneMKL and fftw: " << diff << std::endl;

    return 0;
}