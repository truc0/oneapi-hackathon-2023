#include "oneapi/mkl.hpp"
#include "generator.h"
#include "executor.h"

const unsigned N = 2048;

// std::vector<std::complex<float>, RealAllocatorType> compute_sycl(unsigned N, sycl::queue &queue, std::vector<float, RealAllocatorType> src)
// {
//     oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
//                                  oneapi::mkl::dft::domain::REAL>
//         desc(N);

//     desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
//     desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
//     desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, 1);
//     desc.commit(queue);

//     std::vector<std::complex<float>, RealAllocatorType> dst(N);

//     auto event = oneapi::mkl::dft::compute_forward<
//         decltype(desc), float, std::complex<float>>(desc, src.data(), dst.data());
//     event.wait();

//     return dst;
// }

int main()
{
    auto generator = Generator();
    auto srcReal = generator.generate2DRandomArray(N);
    // auto srcImag = generator.generate2DZeroArray(N);

    FFTExecutor *executor = new OneMKLFFTExecutor();
    auto dst = executor->execute(N, srcReal, generator.queue());

    for (const auto &it : dst)
    {
        std::cout << it.real() << '\t' << it.imag() << std::endl;
    }

    return 0;
}