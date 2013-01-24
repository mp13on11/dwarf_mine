#include "BenchmarkRunner.h"
#include "Configuration.h"
#include "MpiHelper.h"
#include "SchedulerFactory.h"

using namespace std;
using namespace std::chrono;

/**
 * BenchmarkRunner determines the available devices and benchmarks them idenpendently
 */
BenchmarkRunner::BenchmarkRunner(const Configuration& config) :
        _iterations(config.iterations()), _warmUps(config.warmUps()),
        problemStatement(config.createProblemStatement(true)),
        scheduler(config.createScheduler())
{
    for (size_t i = 0; i < MpiHelper::numberOfNodes(); ++i)
        _nodesets.push_back({{static_cast<NodeId>(i), 1}});
}

/**
 * BenchmarkRunner uses the given (weighted) result to benchmark the nodeset as cluster
 */
BenchmarkRunner::BenchmarkRunner(const Configuration& config, const BenchmarkResult& result) :
        _iterations(config.iterations()), _warmUps(config.warmUps()),
        problemStatement(config.createProblemStatement(false)),
        scheduler(config.createScheduler())
{
    _nodesets.push_back(result);
}

void BenchmarkRunner::runBenchmark()
{
    if (MpiHelper::isMaster())
    {
        for (size_t nodeset = 0; nodeset < _nodesets.size(); ++nodeset)
        {
            scheduler->setNodeset(_nodesets[nodeset]);
            _timedResults[nodeset] = benchmarkNodeset();
        }
        weightTimedResults();
    }
    else
    {
        getBenchmarked();
    }
}

unsigned int BenchmarkRunner::benchmarkNodeset()
{
    scheduler->provideData(*problemStatement);
    for (size_t i = 0; i < _warmUps; ++i)
    {
        measureCall();
    }
    chrono::microseconds sum(0);
    for (size_t i = 0; i < _iterations; ++i)
    {
        sum += measureCall();
    }
    scheduler->outputData(*problemStatement);
    return (sum / _iterations).count();
}

void BenchmarkRunner::getBenchmarked()
{
    for (size_t i = 0; i < _iterations + _warmUps; ++i)
        scheduler->dispatch(); // slave side
}

microseconds BenchmarkRunner::measureCall()
{
    high_resolution_clock::time_point before = high_resolution_clock::now();
    scheduler->dispatch();
    return high_resolution_clock::now() - before;
}

void BenchmarkRunner::weightTimedResults()
{
    int runtimeSum = 0;
    for (const auto& nodeResult : _timedResults)
    {
        runtimeSum += nodeResult.second;
    }
    for (const auto& nodeResult : _timedResults)
    {
        _weightedResults[nodeResult.first] = (nodeResult.second * 1.0 / runtimeSum) * 100;
    }
}
