#include <chrono>
#include <fftw3.h>
#include "oneapi/mkl.hpp"
#include "generator.h"
#include "executor.h"
#include "utils.h"

const unsigned N = 2048;
const unsigned size = N * N;
const unsigned validCol = N / 2 + 1;

int main()
{
    auto generator = Generator();
    auto input = generator.generate2DRandomArray(N);

#ifdef DEBUG
    printArray(input);
#endif

    OneMKLFFTExecutor oneExecutor;
    ComplexFloatVector oneResult(size, generator.allocator());
    std::chrono::nanoseconds oneTime;

    SimpleFFTExecutor simpleExecutor;
    ComplexFloatVector simpleResult(size, generator.allocator());
    std::chrono::nanoseconds simpleTime;

    FFTWFFTExecutor fftwExecutor;
    ComplexFloatVector fftwResult(size, generator.allocator());
    std::chrono::nanoseconds fftwTime;

    {
        auto t1 = std::chrono::high_resolution_clock::now();
        oneResult = oneExecutor.execute(N, input, generator.queue());
        auto t2 = std::chrono::high_resolution_clock::now();
        oneTime = t2 - t1;
    }

#ifdef ENABLE_SIMPLE
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        simpleResult = simpleExecutor.execute(N, input, generator.queue());
        auto t2 = std::chrono::high_resolution_clock::now();
        simpleTime = t2 - t1;
    }
#endif

    {
        auto t1 = std::chrono::high_resolution_clock::now();
        fftwResult = fftwExecutor.execute(N, input, generator.queue());
        auto t2 = std::chrono::high_resolution_clock::now();
        fftwTime = t2 - t1;
    }

#ifdef DEBUG
    std::cout << "OneAPI" << std::endl;
    printArray(oneResult);
    std::cout << "FFTW" << std::endl;
    printArray(fftwResult);
#ifdef ENABLE_SIMPLE
    std::cout << "Simple" << std::endl;
    printArray(simpleResult);
#endif
#endif

    std::cout << "OneMKLFFTExecutor time: " << oneTime.count() << " ns" << std::endl;
    std::cout << "FFTWFFTExecutor time: " << fftwTime.count() << " ns" << std::endl;
#ifdef ENABLE_SIMPLE
    std::cout << "SimpleFFTExecutor time: " << simpleTime.count() << " ns" << std::endl;
#endif

    double diffOne = diffArray(oneResult, fftwResult, N, validCol);
#ifdef ENABLE_SIMPLE
    double diffSimple = diffArray(simpleResult, fftwResult, N, validCol);
    double diffOneAndSimple = diffArray(oneResult, simpleResult, N, validCol);
#endif

    std::cout << "Difference of oneMKL and fftw: " << diffOne << std::endl;
#ifdef ENABLE_SIMPLE
    std::cout << "Difference of simple and fftw: " << diffSimple << std::endl;
    std::cout << "Difference of oneMKL and simple: " << diffOneAndSimple << std::endl;
#endif

    return 0;
}