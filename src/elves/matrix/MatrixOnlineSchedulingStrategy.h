#pragma once

#include "MatrixSlicer.h"
#include "common/BenchmarkResults.h"

#include <vector>

class MatrixOnlineScheduler;

class MatrixOnlineSchedulingStrategy
{
public:
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet) = 0;
};
