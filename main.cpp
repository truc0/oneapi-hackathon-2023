#include <chrono>
#include <fftw3.h>
#include "oneapi/mkl.hpp"
#include "generator.h"
#include "executor.h"

// const unsigned N = 2048;
const unsigned N = 4;
const unsigned size = N * N;

ComplexFloatVector use_fftw(const RealFloatVector &input, const ComplexAllocatorType &allocator)
{
    // std::vector<std::complex<float>> in(input.begin(), input.end());
    auto in = fftwf_alloc_complex(size);
    auto out = fftwf_alloc_complex(size);
    for (unsigned i = 0; i < input.size(); ++i)
    {
        in[i][0] = input[i];
        in[i][1] = 0;
    }

    auto plan = fftwf_plan_dft_2d(N, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    auto t1 = std::chrono::high_resolution_clock::now();
    fftwf_execute(plan);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto time = t2 - t1;
    std::cout << "FFTW time: " << time.count() << " ns" << std::endl;

    ComplexFloatVector output(size, allocator);
    for (unsigned i = 0; i < size; ++i)
    {
        output[i] = {out[i][0], out[i][1]};
    }

    fftwf_free(in);
    fftwf_free(out);
    return output;
}

int main()
{
    auto generator = Generator();
    auto input = generator.generate2DRandomArray(N);

    OneMKLFFTExecutor oneExecutor;
    ComplexFloatVector oneResult(size, generator.allocator());
    std::chrono::nanoseconds oneTime;

    SimpleFFTExecutor simpleExecutor;
    ComplexFloatVector simpleResult(size, generator.allocator());
    std::chrono::nanoseconds simpleTime;

    {
        auto t1 = std::chrono::high_resolution_clock::now();
        oneResult = oneExecutor.execute(N, input, generator.queue());
        auto t2 = std::chrono::high_resolution_clock::now();

        oneTime = t2 - t1;
    }

    // {
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     simpleResult = simpleExecutor.execute(N, input, generator.queue());
    //     auto t2 = std::chrono::high_resolution_clock::now();

    //     simpleTime = t2 - t1;
    // }

    auto refResult = use_fftw(input, generator.allocator());

    std::cout << "OneMKLFFTExecutor time: " << oneTime.count() << " ns" << std::endl;
    std::cout << "SimpleFFTExecutor time: " << simpleTime.count() << " ns" << std::endl;

    {
        long double diff = 0;
        for (unsigned i = 0; i < size; ++i)
        {
            diff += std::abs(refResult[i] - simpleResult[i]);
        }
        std::cout << "Difference of simple: " << diff << std::endl;
    }

    {
        long double diff = 0;
        for (unsigned i = 0; i < size; ++i)
        {
            diff += std::abs(refResult[i] - oneResult[i]);
        }
        std::cout << "Difference of one: " << diff << std::endl;
    }

    return 0;
}