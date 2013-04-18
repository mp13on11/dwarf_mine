#include "TimingProfiler.h"

#include <functional>
#include <numeric>

using namespace std;
using namespace std::chrono;

microseconds TimingProfiler::averageIterationTime() const
{
    if (measurements.empty())
        return microseconds(0);
    
    return accumulate(
        measurements.begin(), measurements.end(), microseconds(0), plus<microseconds>()
    ) / measurements.size();
}

vector<microseconds> TimingProfiler::iterationTimes() const
{
    return measurements;
}

void TimingProfiler::beginIterationBlock()
{
    measurements.clear();
}

void TimingProfiler::beginIteration()
{
    startOfIteration = clock::now();
}

void TimingProfiler::endIteration()
{
    measurements.push_back(durationOfIteration());
}

void TimingProfiler::endIterationBlock()
{
}

microseconds TimingProfiler::durationOfIteration() const
{
    return duration_cast<microseconds>(clock::now() - startOfIteration);
}