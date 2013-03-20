#pragma once

#include "MatrixSlicer.h"
#include "common/BenchmarkResults.h"

#include <vector>
#include <map>
#include <memory>

class MatrixOnlineScheduler;

class MatrixOnlineSchedulingStrategy
{
public:
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet) = 0;
    virtual int getLastWorkAmountFor(
        const MatrixOnlineScheduler& scheduler,
        const int node) = 0;
    virtual int getNextWorkAmountFor(
        const MatrixOnlineScheduler& scheduler,
        const int node) = 0;

protected:
    virtual void reset() = 0;
};

