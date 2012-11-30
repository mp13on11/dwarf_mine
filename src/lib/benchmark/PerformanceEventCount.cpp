#include "benchmark/PerformanceEventCount.h"
#include <limits>

PerformanceEventCount::PerformanceEventCount(PerformanceEvent* performanceEvent)
: performanceEvent(performanceEvent)
{
    resetSoft();
}

PerformanceEventCount::~PerformanceEventCount()
{
    delete performanceEvent;
}

void PerformanceEventCount::reset()
{
    resetHard();
}

void PerformanceEventCount::resetHard()
{
    resetSoft();
    counts.clear();
}

void PerformanceEventCount::resetSoft()
{
    min = std::numeric_limits<count_t>::max();
    max = std::numeric_limits<count_t>::min();
    avg = 0;
}

void PerformanceEventCount::evaluate()
{
    resetSoft();
    processCounts();
}

void PerformanceEventCount::processCounts()
{
    count_t sum = 0;
    for (count_t& count : counts)
    {
        sum += count;
        if (count < min) min = count;
        if (count > max) max = count;
    }
    avg = sum/counts.size();
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
