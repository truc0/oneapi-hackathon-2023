#pragma once

#include <iostream>
#include <vector>
#include "generator.h"

using RealFloatVector = std::vector<float, RealAllocatorType>;
using ComplexFloatVector = std::vector<std::complex<float>, ComplexAllocatorType>;

/**
 * FFTExecutor is an abstract class for executing FFT algorithms.
 */
class FFTExecutor
{
public:
    virtual ComplexFloatVector execute(const unsigned N, RealFloatVector &input, sycl::queue &queue) = 0;
};

class SimpleFFTExecutor : public FFTExecutor
{
public:
    ComplexFloatVector execute(const unsigned N, RealFloatVector &input, sycl::queue &queue) override;
};

class OneMKLFFTExecutor : public FFTExecutor
{
public:
    ComplexFloatVector execute(const unsigned N, RealFloatVector &input, sycl::queue &queue) override;
};