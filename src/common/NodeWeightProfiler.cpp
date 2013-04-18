/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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