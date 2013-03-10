#pragma once

#include "MatrixOnlineSchedulingStrategy.h"
#include "MatrixSlicerOnline.h"

class MatrixOnlineSchedulingRowwise : public MatrixOnlineSchedulingStrategy
{
public:
    MatrixOnlineSchedulingRowwise(MatrixOnlineScheduler& scheduler);
    virtual ~MatrixOnlineSchedulingRowwise();
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet);
    virtual int getWorkAmountFor(const NodeId node);

private:
    MatrixSlicerOnline slicer;
};

