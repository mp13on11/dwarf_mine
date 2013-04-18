#pragma once

#include "Profiler.h"

#include <chrono>
#include <vector>

class TimingProfiler : public Profiler
{
public:
    std::chrono::microseconds averageIterationTime() const;
    std::vector<std::chrono::microseconds> iterationTimes() const;

    virtual void beginIterationBlock();
    virtual void beginIteration();
    virtual void endIteration();
    virtual void endIterationBlock();

private:
    typedef std::chrono::steady_clock clock;

    std::vector<std::chrono::microseconds> measurements;
    clock::time_point startOfIteration;

    std::chrono::microseconds durationOfIteration() const;
};