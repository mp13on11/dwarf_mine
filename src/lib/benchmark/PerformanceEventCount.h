#pragma once

#include "benchmark/PerformanceEvent.h"
#include <vector>

using count_t = long long;
using counts_t = std::vector<count_t>;

class PerformanceEventCount
{
private:
    PerformanceEvent* performanceEvent;
    counts_t counts;
    count_t min;
    count_t max;
    count_t avg;
public:
    PerformanceEventCount(PerformanceEvent* performanceEvent);
    ~PerformanceEventCount();
    void reset();
    void processCounts();
    const PerformanceEvent& getPerformanceEvent() const;
    count_t getMinimum() const;
    count_t getMaximum() const;
    count_t getAverage() const;
};
