#include <iostream>
#include "oneapi/mkl.hpp"

#include "executor.h"

/* SimpleExecutor */

ComplexFloatVector SimpleFFTExecutor::execute(const unsigned N, RealFloatVector &input, sycl::queue &queue)
{
    std::cout << "SimpleFFTExecutor::run()" << std::endl;
}

/* OneMKLFFTExecutor */

ComplexFloatVector OneMKLFFTExecutor::execute(const unsigned N, RealFloatVector &input, sycl::queue &queue)
{
    std::cout << "OneMKLFFTExecutor::execute()" << std::endl;

    const auto size = N * N;
    auto allocator = ComplexAllocatorType(queue.get_context(), queue.get_device());

    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                 oneapi::mkl::dft::domain::REAL>
        desc({N, N});

    // desc.set_value(oneapi::mkl::dft::config_param::DIMENSION, static_cast<std::int64_t>(2));
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, static_cast<std::int64_t>(1));
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    desc.commit(queue);

    ComplexFloatVector dst(size, allocator);

    auto event = oneapi::mkl::dft::compute_forward<
        decltype(desc), float, std::complex<float>>(desc, input.data(), dst.data());
    event.wait();

    return dst;
}