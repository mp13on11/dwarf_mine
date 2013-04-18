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

#include "BenchmarkRunner.h"
#include "Communicator.h"
#include "Configuration.h"
#include "Profiler.h"
#include "Scheduler.h"

using namespace std;

BenchmarkRunner::BenchmarkRunner(const Configuration& config) :
        config(&config),
        iterations(config.iterations()), warmUps(config.warmUps()),
        fileProblem(config.createProblemStatement()),
        generatedProblem(config.createGeneratedProblemStatement())
{
}

void BenchmarkRunner::runBenchmark(const Communicator& communicator, Profiler& profiler) const
{
    runBenchmarkInternal(communicator, profiler, fileProblem);
}

void BenchmarkRunner::runPreBenchmark(const Communicator& communicator, Profiler& profiler) const
{
    runBenchmarkInternal(communicator, profiler, generatedProblem);
}

void BenchmarkRunner::runBenchmarkInternal(
    const Communicator& communicator, 
    Profiler& profiler,
    const unique_ptr<ProblemStatement>& problem
) const
{   
    if (!communicator.isValid())
        return;
    
    unique_ptr<Scheduler> scheduler = config->createScheduler(communicator);

    if (communicator.isMaster())
    {
        scheduler->provideData(*problem);
        run(*scheduler, profiler);
        scheduler->outputData(*problem);
    }
    else
    {
        run(*scheduler, profiler);
    }
}

void BenchmarkRunner::run(Scheduler& scheduler, Profiler& profiler) const
{
    for (size_t i = 0; i < warmUps; ++i)
    {
        scheduler.dispatch();
    }

    profiler.beginIterationBlock();

    for (size_t i = 0; i < iterations; ++i)
    {
        profiler.beginIteration();
        scheduler.dispatch();
        profiler.endIteration();
    }

    profiler.endIterationBlock();
