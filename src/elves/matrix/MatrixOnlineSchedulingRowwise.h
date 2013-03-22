#pragma once

#include "MatrixOnlineSchedulingStrategy.h"
#include "MatrixSlicerOnline.h"

#include <map>

class MatrixOnlineSchedulingRowwise : public MatrixOnlineSchedulingStrategy
{
public:
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet);
    virtual int getLastWorkAmountFor(
        const MatrixOnlineScheduler& scheduler,
        const int node);
    virtual int getNextWorkAmountFor(
        const MatrixOnlineScheduler& scheduler,
        const int node);

protected:
    virtual void reset();

private:
    static const int defaultWorkAmount;
    MatrixSlicerOnline slicer;
    std::map<int, int> lastWorkAmounts;
};
