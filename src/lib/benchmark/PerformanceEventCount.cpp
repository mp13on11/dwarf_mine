#include "benchmark/PerformanceEventCount.h"
#include <limits>

PerformanceEventCount::PerformanceEventCount(PerformanceEvent* performanceEvent)
: performanceEvent(performanceEvent)
{
    reset();
}

PerformanceEventCount::~PerformanceEventCount()
{
    delete performanceEvent;
}

void PerformanceEventCount::reset()
{
    counts.clear();
    min = std::numeric_limits<long long>::max();
    max = std::numeric_limits<long long>::min();
    avg = 0;
}

void PerformanceEventCount::processCounts()
{
    count_t avgSum = 0;
    for (count_t& count : counts)
    {
        avgSum += count;
        if (count < min) min = count;
        if (count > max) max = count;
    }
    avg = avgSum = counts.size();
}

const PerformanceEvent& PerformanceEventCount::getPerformanceEvent() const
{
    return *performanceEvent;
}

count_t PerformanceEventCount::getMinimum() const
{
    return min;
}

count_t PerformanceEventCount::getMaximum() const
{
    return max;
}

count_t PerformanceEventCount::getAverage() const
{
    return avg;
}
