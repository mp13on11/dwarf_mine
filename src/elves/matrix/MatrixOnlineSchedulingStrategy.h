#pragma once

#include <cstddef>
#include <vector>

template <typename T>
class Matrix;
class MatrixSlice;
class BenchmarkResult;

class MatrixOnlineSchedulingStrategy
{
public:
    virtual size_t getWorkQueueSize() = 0;
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet) = 0;
};
