#include "BenchmarkRunner.h"
#include "Configuration.h"
#include "MpiHelper.h"
#include "SchedulerFactory.h"


using namespace std;

/**
 * BenchmarkRunner determines the available devices and benchmarks them idenpendently
 */
BenchmarkRunner::BenchmarkRunner(const Configuration& config) :
        _iterations(config.iterations()), _warmUps(config.warmUps()),
        problemStatement(config.createProblemStatement(true))
{
    for (size_t i = 0; i < MpiHelper::numberOfNodes(); ++i)
        _nodesets.push_back({{static_cast<NodeId>(i), 1}});

    auto factory = config.createSchedulerFactory();
    scheduler = factory->createScheduler();
}

/**
 * BenchmarkRunner uses the given (weighted) result to benchmark the nodeset as cluster
 */
BenchmarkRunner::BenchmarkRunner(const Configuration& config, const BenchmarkResult& result) :
        _iterations(config.iterations()), _warmUps(config.warmUps()),
        problemStatement(config.createProblemStatement(false))
{
    _nodesets.push_back(result);

    auto factory = config.createSchedulerFactory();
    scheduler = factory->createScheduler();
}

std::chrono::microseconds BenchmarkRunner::measureCall(Scheduler& scheduler) {
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    scheduler.dispatch();
    return clock::now() - before;
}

unsigned int BenchmarkRunner::benchmarkNodeset()
{
    scheduler->provideData(*problemStatement);
    for (size_t i = 0; i < _warmUps; ++i)
    {
        measureCall(*scheduler);
    }
    chrono::microseconds sum(0);
    for (size_t i = 0; i < _iterations; ++i)
    {
        sum += measureCall(*scheduler);
    }
    scheduler->outputData(*problemStatement);
    return (sum / _iterations).count();
}

void BenchmarkRunner::getBenchmarked()
{
    for (size_t i = 0; i < _iterations + _warmUps; ++i)
        scheduler->dispatch(); // slave side
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
