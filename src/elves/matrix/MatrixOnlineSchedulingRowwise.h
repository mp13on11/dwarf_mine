#pragma once

#include "MatrixOnlineSchedulingStrategy.h"
#include "MatrixSlicerOnline.h"

class MatrixOnlineSchedulingRowwise : public MatrixOnlineSchedulingStrategy
{
public:
    virtual size_t getWorkQueueSize();
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet);

private:
    MatrixSlicerOnline slicer;
};
