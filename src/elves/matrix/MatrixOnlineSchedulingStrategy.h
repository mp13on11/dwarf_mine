#pragma once

#include "MatrixOnlineScheduler.h"
#include "MatrixSlicer.h"

#include <vector>
#include <map>

typedef int NodeId;
typedef double Rating;
typedef std::pair<NodeId, Rating> NodeRating;
typedef std::map<NodeId, Rating> BenchmarkResult;

class MatrixOnlineSchedulingStrategy
{
public:
    MatrixOnlineSchedulingStrategy(MatrixOnlineScheduler& scheduler)
    {
        this->scheduler = std::shared_ptr<MatrixOnlineScheduler>(&scheduler);
    }

    virtual ~MatrixOnlineSchedulingStrategy()
    {
    }

    virtual std::vector<MatrixSlice> getSliceDefinitions(
        const Matrix<float>& result,
        const BenchmarkResult& nodeSet) = 0;

    virtual int getWorkAmountFor(const NodeId node) = 0;

protected:
    MatrixOnlineScheduler& getScheduler()
    {
        return *scheduler;
    }

private:
    std::shared_ptr<MatrixOnlineScheduler> scheduler;
};

