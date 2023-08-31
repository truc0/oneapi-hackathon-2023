#pragma once
#include <vector>

template <class T, class Allocator>
inline void printArray(const std::vector<T, Allocator> &array)
{
    for (const auto item : array)
    {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

template <class T, class Allocator>
inline double diffArray(const std::vector<T, Allocator> &array1, const std::vector<T, Allocator> &array2, const unsigned N, const unsigned validCol)
{
    double diff = 0;
    for (unsigned row = 0; row < N; ++row)
    {
        for (unsigned col = 0; col < validCol; ++col)
        {
            diff += std::abs(array1[row * N + col] - array2[row * N + col]);
        }
    }
    return diff;
}