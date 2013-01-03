
#include <mpi.h>
#include <vector>
#include <memory>
#include <cmath>
#include "Elf.h"
#include "BenchmarkRunner.h"
#include "Scheduler.h"

using namespace std;

const size_t WARMUP_ITERATIONS = 50;

/**
 * BenchmarkRunner determines the available devices and benchmarks them idenpendently
 */
BenchmarkRunner::BenchmarkRunner(size_t iterations)
    : _iterations(iterations), _warmUps(WARMUP_ITERATIONS)
{
    for (NodeId i = 0; i < MPI::COMM_WORLD.Get_size(); ++i)
        _nodesets.push_back({{i, 0}});
}

/**
 * BenchmarkRunner uses the given (weighted) result to benchmark the nodeset as cluster
 */
BenchmarkRunner::BenchmarkRunner(Configuration& config, const BenchmarkResult& result)
    : _iterations(config.getNumberOfIterations()), _warmUps(config.getNumberOfWarmUps())
{
    _nodesets.push_back(result);
}

std::chrono::microseconds BenchmarkRunner::measureCall(ProblemStatement& statement, Scheduler& scheduler) {
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    scheduler.dispatch(statement);
    return clock::now() - before;
}

unsigned int BenchmarkRunner::benchmarkNodeset(ProblemStatement& statement, Scheduler& scheduler)
{
    for (size_t i = 0; i < _warmUps; ++i)
    {
        measureCall(statement, scheduler);
    }
    chrono::microseconds sum(0);
    for (size_t i = 0; i < _iterations; ++i)
    {
        sum += measureCall(statement, scheduler);
    }
    return (sum / _iterations).count();
}

void BenchmarkRunner::getBenchmarked(ProblemStatement& statement, Scheduler& scheduler)
{
    for (size_t i = 0; i < _iterations + _warmUps; ++i)
        scheduler.dispatch(statement); // slave side
}

void BenchmarkRunner::runBenchmark(ProblemStatement& statement, const ElfFactory& factory)
{
    unique_ptr<Scheduler> scheduler = factory.createScheduler();

    unique_ptr<Elf> elf = factory.createElf();
    scheduler->setElf(elf.get());

    if (MPI::COMM_WORLD.Get_rank() == MASTER)
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
        getBenchmarked(statement, *scheduler);
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
        _weightedResults[nodeResult.first] = 100 - nodeResult.second * 100 / runtimeSum;
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
