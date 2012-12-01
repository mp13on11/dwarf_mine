#pragma once

#include "benchmark/PerformanceEvent.h"
#include <vector>

typedef long long count_t;
typedef std::vector<count_t> counts_t;

class PerformanceEventCount
{
public:
    PerformanceEventCount(PerformanceEvent* performanceEvent);
    ~PerformanceEventCount();
    void reset();
    void evaluate();
    const PerformanceEvent& getPerformanceEvent() const;
    count_t getMinimum() const;
    count_t getMaximum() const;
    count_t getAverage() const;

private:
    PerformanceEvent* performanceEvent;
    counts_t counts;
    count_t min;
    count_t max;
    count_t avg;
    void resetHard();
    void resetSoft();
    void processCounts();
};
