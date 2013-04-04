#include "NodeWeightProfiler.h"

#include <algorithm>
#include <iostream>
#include <numeric>

using namespace std;
using namespace std::chrono;

NodeWeightProfiler::NodeWeightProfiler()
{
    cout << "\tAverage execution times (microseconds):" << endl;
}

void NodeWeightProfiler::saveExecutionTime()
{
    averageExecutionTimes.push_back(
            averageIterationTime()
        );

    cout << "\t\tRank " << averageExecutionTimes.size() - 1 
            << ":\t" << averageExecutionTimes.back().count() << endl;
}

vector<double> NodeWeightProfiler::nodeWeights() const
{
    vector<double> weights = unscaledNodeWeights();
    double fullWeight = accumulate(weights.begin(), weights.end(), 0.0);

    cout << "\tNode weights (scaled):" << endl;
    for (size_t i=0; i<weights.size(); ++i)
    {
        weights[i] /= fullWeight;
        cout << "\t\tRank " << i << ":\t" << weights[i] << endl;
    }

    return weights;
}

vector<double> NodeWeightProfiler::unscaledNodeWeights() const
{
    if (averageExecutionTimes.empty())
        return vector<double>();

    microseconds max = *max_element(averageExecutionTimes.begin(), averageExecutionTimes.end());
    vector<double> weights;

    for (const microseconds& time : averageExecutionTimes)
    {
        double weight = static_cast<double>(max.count()) / static_cast<double>(time.count());
        weights.push_back(weight);
    }

    return weights;
}