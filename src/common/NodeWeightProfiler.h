#pragma once

#include "TimingProfiler.h"

#include <chrono>
#include <vector>

class NodeWeightProfiler : public TimingProfiler
{
public:
    NodeWeightProfiler();

    void saveExecutionTime();
    std::vector<double> nodeWeights() const;

private:
    std::vector<std::chrono::microseconds> averageExecutionTimes;

    std::vector<double> unscaledNodeWeights() const;
};