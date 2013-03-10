#pragma once

#include <vector>

class MatrixOnlineScheduler;
class MatrixSlice;
template <typename T>
class Matrix<T>;
class BenchmarkResult;

class MatrixOnlineSchedulingStrategy
{
public:
    MatrixOnlineSchedulingStrategy(MatrixOnlineScheduler scheduler)
    : scheduler(scheduler)
    {
    }

    virtual ~MatrixOnlineSchedulingStrategy()
    {
    }

    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet) = 0;

    virtual int getWorkAmountFor(const NodeId node) = 0;

private:
    MatrixOnlineScheduler scheduler;
};

