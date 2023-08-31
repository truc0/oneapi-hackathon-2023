#include <iostream>
#include <algorithm>
#include <chrono>
#include <fftw3.h>
#include "oneapi/mkl.hpp"
#include "generator.h"
#include "executor.h"
#include "utils.h"

const unsigned N = 2048;
const unsigned size = N * N;
const unsigned validCol = N / 2 + 1;
const unsigned iterationsCnt = 1024;

long double run_once();
void run_multiple();

int main()
{
    // run_once();
    run_multiple();
    return 0;
}

void run_multiple()
{
    auto generator = Generator();

    OneMKLFFTExecutor oneExecutor;
    ComplexFloatVector oneResult(size, generator.allocator());
    std::vector<std::chrono::nanoseconds> oneTime;

    FFTWFFTExecutor fftwExecutor;
    ComplexFloatVector fftwResult(size, generator.allocator());
    std::vector<std::chrono::nanoseconds> fftwTime;

    std::vector<long double> diff;

    for (unsigned i = 0; i < iterationsCnt; ++i)
    {
        auto input = generator.generate2DRandomArray(N);

        auto tstart = std::chrono::high_resolution_clock::now();
        oneResult = oneExecutor.execute(N, input, generator.queue());
        auto tend = std::chrono::high_resolution_clock::now();
        oneTime.push_back(tend - tstart);

        tstart = std::chrono::high_resolution_clock::now();
        fftwResult = fftwExecutor.execute(N, input, generator.queue());
        tend = std::chrono::high_resolution_clock::now();
        fftwTime.push_back(tend - tstart);

        diff.push_back(diffArray(oneResult, fftwResult, N, validCol));

        std::cout << '\r' << i + 1 << '/' << iterationsCnt << std::flush;
    }
    std::cout << " finished." << std::endl;

    /* Analysis */
    // diff
    long double diffTotal = 0;
    long double diffMax = 0;
    long double diffMin = 999999;
    for (const auto item : diff)
    {
        diffMax = std::max(item, diffMax);
        diffMin = std::min(item, diffMin);
        diffTotal += item;
    }

    // time
    std::chrono::nanoseconds oneTimeTotal = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds oneTimeMax = std::chrono::nanoseconds::min();
    std::chrono::nanoseconds oneTimeMin = std::chrono::nanoseconds::max();
    std::chrono::nanoseconds fftwTimeTotal = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds fftwTimeMax = std::chrono::nanoseconds::min();
    std::chrono::nanoseconds fftwTimeMin = std::chrono::nanoseconds::max();
    for (const auto item : oneTime)
    {
        oneTimeMax = std::max(item, oneTimeMax);
        oneTimeMin = std::min(item, oneTimeMin);
        oneTimeTotal += item;
    }
    for (const auto item : fftwTime)
    {
        fftwTimeMax = std::max(item, fftwTimeMax);
        fftwTimeMin = std::min(item, fftwTimeMin);
        fftwTimeTotal += item;
    }

    // output
    std::cout << "OneMKLFFTExecutor time" << std::endl
              << "===============================" << std::endl
              << "Avg:\t" << oneTimeTotal.count() / iterationsCnt << " ns" << std::endl
              << "Max:\t" << oneTimeMax.count() << " ns" << std::endl
              << "Min:\t" << oneTimeMin.count() << " ns" << std::endl
              << std::endl;
    std::cout << "FFTWFFTExecutor time" << std::endl
              << "===============================" << std::endl
              << "Avg:\t" << fftwTimeTotal.count() / iterationsCnt << " ns" << std::endl
              << "Max:\t" << fftwTimeMax.count() << " ns" << std::endl
              << "Min:\t" << fftwTimeMin.count() << " ns" << std::endl
              << std::endl;

    std::cout << "Difference" << std::endl
              << "===============================" << std::endl
              << "Avg difference: " << diffTotal / iterationsCnt << std::endl
              << "Max difference: " << diffMax << std::endl
              << "Min difference: " << diffMin << std::endl;
}

long double run_once()
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

    auto diffOne = diffArray(oneResult, fftwResult, N, validCol);
#ifdef ENABLE_SIMPLE
    auto diffSimple = diffArray(simpleResult, fftwResult, N, validCol);
    auto diffOneAndSimple = diffArray(oneResult, simpleResult, N, validCol);
#endif

    std::cout << "Difference of oneMKL and fftw: " << diffOne << std::endl;
#ifdef ENABLE_SIMPLE
    std::cout << "Difference of simple and fftw: " << diffSimple << std::endl;
    std::cout << "Difference of oneMKL and simple: " << diffOneAndSimple << std::endl;
#endif

    return diffOne;
}