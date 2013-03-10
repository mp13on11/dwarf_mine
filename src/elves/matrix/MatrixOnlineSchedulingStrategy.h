#pragma once

#include "MatrixSlicer.h"

#include <vector>
#include <map>
#include <memory>

class MatrixOnlineScheduler;
typedef int NodeId;
typedef double Rating;
typedef std::pair<NodeId, Rating> NodeRating;
typedef std::map<NodeId, Rating> BenchmarkResult;

class MatrixOnlineSchedulingStrategy
{
public:
    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet) = 0;
    virtual int getWorkAmountFor(const NodeId node) = 0;

protected:
    virtual void reset() = 0;
};

