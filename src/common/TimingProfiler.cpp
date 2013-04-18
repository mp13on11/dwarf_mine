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