#include <iostream>
#include <vector>
#include "CL/sycl.hpp"
#include "oneapi/mkl/rng.hpp"

#include "random.h"

namespace Random
{
/**
 * Genrerate a N * N 2D matrix using oneKML random
 * number generators (RNGs) API
 */
auto generate2DArray(unsigned N)
{
    const auto size = N * N;
    sycl::queue queue;
    sycl::usm_allocator<double, sycl::usm::alloc::device> allocator(queue);

    std::vector<double, decltype(allocator)> r(size, allocator);
    
    oneapi::mkl::rng::default_engine engine(queue);
    oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::uniform_method::accurate> distr;

    auto event = oneapi::mkl::rng::generate(distr, engine, size, r.data());
    event.wait();

    return r;
}

};
