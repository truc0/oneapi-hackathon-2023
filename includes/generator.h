#pragma once

#include <iostream>
#include <vector>
#include "CL/sycl.hpp"
#include "oneapi/mkl/rng.hpp"

using RealAllocatorType = sycl::usm_allocator<float, sycl::usm::alloc::shared>;
using ComplexAllocatorType = sycl::usm_allocator<std::complex<float>, sycl::usm::alloc::shared>;

class Generator
{
public:
  /* constructors */

  Generator()
      : queue_(sycl::default_selector{}),
        allocator_(queue_.get_context(), queue_.get_device()) {}

  Generator(sycl::queue queue)
      : queue_(queue), allocator_(queue_.get_context(), queue_.get_device()) {}

  /* generator functions */

  std::vector<float, RealAllocatorType> generate2DRandomArray(unsigned N);

  std::vector<float, RealAllocatorType> generate2DZeroArray(unsigned N);

  sycl::queue &queue() { return queue_; }
  RealAllocatorType &allocator() { return allocator_; }

private:
  sycl::queue queue_;
  RealAllocatorType allocator_;
};
