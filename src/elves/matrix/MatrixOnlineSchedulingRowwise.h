#pragma once

#include "MatrixOnlineSchedulingStrategy.h"
#include "MatrixSlicerOnline.h"

class MatrixOnlineSchedulingRowwise : public MatrixOnlineSchedulingStrategy
{
public:
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet);
    virtual int getWorkAmountFor(const int node);

protected:
    virtual void reset();

private:
    MatrixSlicerOnline slicer;
};

