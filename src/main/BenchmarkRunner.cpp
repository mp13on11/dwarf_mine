#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "BenchmarkRunner.h"
#include "Elf.h"
#include "MpiHelper.h"
#include "Scheduler.h"

using namespace std;

const size_t WARMUP_ITERATIONS = 50;

/**
 * BenchmarkRunner determines the available devices and benchmarks them idenpendently
 */
BenchmarkRunner::BenchmarkRunner(Configuration& config)
    : _iterations(config.getNumberOfIterations()), _warmUps(config.getNumberOfWarmUps())
{
    for (size_t i = 0; i < MpiHelper::numberOfNodes(); ++i)
        _nodesets.push_back({{static_cast<NodeId>(i), 0}});
}

/**
 * BenchmarkRunner uses the given (weighted) result to benchmark the nodeset as cluster
 */
BenchmarkRunner::BenchmarkRunner(Configuration& config, const BenchmarkResult& result)
    : _iterations(config.getNumberOfIterations()), _warmUps(config.getNumberOfWarmUps())
{
    _nodesets.push_back(result);
}

std::chrono::microseconds BenchmarkRunner::measureCall(Scheduler& scheduler) {
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    scheduler.dispatch();
    return clock::now() - before;
}

unsigned int BenchmarkRunner::benchmarkNodeset(ProblemStatement& statement, Scheduler& scheduler)
{
    scheduler.provideData(statement);
    for (size_t i = 0; i < _warmUps; ++i)
    {
        measureCall(scheduler);
    }
    chrono::microseconds sum(0);
    for (size_t i = 0; i < _iterations; ++i)
    {
        sum += measureCall(scheduler);
    }
    scheduler.outputData(statement);
    return (sum / _iterations).count();
}

void BenchmarkRunner::getBenchmarked(Scheduler& scheduler)
{
    for (size_t i = 0; i < _iterations + _warmUps; ++i)
        scheduler.dispatch(); // slave side
}

void BenchmarkRunner::runBenchmark(ProblemStatement& statement, const ElfFactory& factory)
{
    unique_ptr<Scheduler> scheduler = factory.createScheduler();

    if (MpiHelper::isMaster())
    {
        for (size_t nodeset = 0; nodeset < _nodesets.size(); ++nodeset)
        {
            scheduler->setNodeset(_nodesets[nodeset]);
            _timedResults[nodeset] = benchmarkNodeset(statement, *scheduler);
        }
        weightTimedResults();
    }
    else
    {
        getBenchmarked(*scheduler);
    }
}

void BenchmarkRunner::weightTimedResults()
{
    int runtimeSum = 0;
    for (const auto& nodeResult : _timedResults)
    {
        runtimeSum += nodeResult.second;
    }
    for (const auto& nodeResult :  _timedResults)
    {
        _weightedResults[nodeResult.first] = (nodeResult.second * 1.0 / runtimeSum) * 100;
    }
}

BenchmarkResult BenchmarkRunner::getWeightedResults()
{
    return _weightedResults;
}

BenchmarkResult BenchmarkRunner::getTimedResults()
{
    return _timedResults;
}
