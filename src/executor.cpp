#include <iostream>
#include <fftw3.h>
#include "oneapi/mkl.hpp"

#include "executor.h"

/* SimpleExecutor */

ComplexFloatVector SimpleFFTExecutor::execute(const unsigned N, RealFloatVector &input, sycl::queue &queue)
{
    // get logN
    unsigned logN = 0;
    {
        unsigned n = N;
        while (n >>= 1)
        {
            logN++;
        }
    }

    ComplexAllocatorType allocator(queue.get_context(), queue.get_device());
    ComplexFloatVector transposedImm(N * N, allocator);
    ComplexFloatVector output(N * N, allocator);

    for (unsigned index = 0; index < N; ++index)
    {
        ComplexFloatVector row(input.begin() + N * index, input.begin() + N * index + N, allocator);
        ComplexFloatVector Frow = fft(logN, row, allocator);
        // transpose
        for (unsigned i = 0; i < N; i++)
        {
            transposedImm[i * N + index] = Frow[i];
        }
    }
    for (unsigned index = 0; index < N; ++index)
    {
        ComplexFloatVector row(transposedImm.begin() + N * index, transposedImm.begin() + N * index + N, allocator);
        ComplexFloatVector Frow = fft(logN, row, allocator);
        // transpose
        for (unsigned i = 0; i < N; i++)
        {
            output[i * N + index] = Frow[i];
        }
    }

    return output;
}

ComplexFloatVector SimpleFFTExecutor::fft(const unsigned logN, const ComplexFloatVector &input, ComplexAllocatorType &allocator)
{
    if (input.size() == 1)
    {
        return input;
    }

    const auto isize = input.size();
    const auto ihalf = isize / 2;

    ComplexFloatVector odd(ihalf, allocator);
    ComplexFloatVector even(ihalf, allocator);

    for (unsigned i = 0; i < ihalf; i++)
    {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    const auto oddOutput = fft(logN - 1, odd, allocator);
    const auto evenOutput = fft(logN - 1, even, allocator);

    ComplexFloatVector output(input.size(), allocator);

    const double ANG = -2 * M_PI / (1 << logN);
    std::complex<float> w(1), wn(std::cos(ANG), std::sin(ANG));

    for (unsigned i = 0; i < ihalf; i++)
    {
        output[i] = evenOutput[i] + w * oddOutput[i];
        output[i + ihalf] = evenOutput[i] - w * oddOutput[i];
        w *= wn;
    }
    return output;
}

/* OneMKLFFTExecutor */

ComplexFloatVector OneMKLFFTExecutor::execute(const unsigned N, RealFloatVector &input, sycl::queue &queue)
{
    const auto size = N * N;
    auto allocator = ComplexAllocatorType(queue.get_context(), queue.get_device());

    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                 oneapi::mkl::dft::domain::REAL>
        desc({N, N});

    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, static_cast<std::int64_t>(1));
    desc.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
    desc.commit(queue);

    ComplexFloatVector dst(size, allocator);

    oneapi::mkl::dft::compute_forward<
        decltype(desc), float, std::complex<float>>(desc, input.data(), dst.data())
        .wait();

    return dst;
}

/* FFTWFFTExecutor */

ComplexFloatVector FFTWFFTExecutor::execute(const unsigned N, RealFloatVector &input, sycl::queue &queue)
{
    const auto size = N * N;
    const auto validCol = N / 2 + 1;
    auto allocator = ComplexAllocatorType(queue.get_context(), queue.get_device());

    auto in = fftwf_alloc_real(size);
    auto out = fftwf_alloc_complex(size);

    for (unsigned i = 0; i < input.size(); ++i)
    {
        in[i] = input[i];
    }

    auto plan = fftwf_plan_dft_r2c_2d(N, N, in, out, FFTW_MEASURE);

    fftwf_execute(plan);

    ComplexFloatVector output(size, allocator);
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
    ComplexFloatVector dst(size, allocator);

    return dst;
}