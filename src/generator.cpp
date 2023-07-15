#include <iostream>
#include <vector>
#include "CL/sycl.hpp"
#include "oneapi/mkl/rng.hpp"

#include "generator.h"

/**
 * Genrerate a N * N 2D matrix using oneKML random
 * number generators (RNGs) API
 */
std::vector<float, RealAllocatorType>
Generator::generate2DRandomArray(unsigned N)
{
    const auto size = N * N;

    std::vector<float, decltype(allocator_)> r(size, allocator_);

    oneapi::mkl::rng::default_engine engine(queue_);
    oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::accurate> distr;

    auto event = oneapi::mkl::rng::generate(distr, engine, size, r.data());
    event.wait();

    return r;
}

/**
 * Generate a N * N 2D matrix filled by zero
 */
std::vector<float, RealAllocatorType>
Generator::generate2DZeroArray(unsigned N)
{
    const auto size = N * N;
    std::vector<float, decltype(allocator_)> r(size, allocator_);
    std::fill(r.begin(), r.end(), 0.0f);

    auto event = queue_.memset(r.data(), 0, size * sizeof(float));
    event.wait();

    return r;
}
