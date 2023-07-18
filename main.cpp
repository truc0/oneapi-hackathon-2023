#include "oneapi/mkl.hpp"
#include "generator.h"
#include "executor.h"

const unsigned N = 2048;

int main()
{
    auto generator = Generator();
    auto input = generator.generate2DRandomArray(N);

    // FFTExecutor *executor = new OneMKLFFTExecutor();
    SimpleFFTExecutor executor;
    auto dst = executor.execute(N, input, generator.queue());

    for (const auto &it : dst)
    {
        std::cout << it.real() << '\t' << it.imag() << std::endl;
    }

    return 0;
}