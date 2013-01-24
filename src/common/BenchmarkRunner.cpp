#include "BenchmarkRunner.h"
#include "Configuration.h"
#include "MpiHelper.h"
#include "SchedulerFactory.h"

using namespace std;
using namespace std::chrono;

typedef BenchmarkRunner::Measurement Measurement;

/**
 * BenchmarkRunner determines the available devices and benchmarks them idenpendently
 */
BenchmarkRunner::BenchmarkRunner(const Configuration& config) :
        iterations(config.iterations()), warmUps(config.warmUps()),
        individualProblem(config.createProblemStatement(true)),
        clusterProblem(config.createProblemStatement(false)),
        scheduler(config.createScheduler())
{
}

BenchmarkResult BenchmarkRunner::benchmarkIndividualNodes()
{
    vector<Measurement> averageRunTimes;

    for (size_t i=0; i<MpiHelper::numberOfNodes(); ++i)
    {
        vector<Measurement> runTimes = runBenchmark(
                {{static_cast<NodeId>(i), 1}}, *individualProblem
            );
        averageRunTimes.push_back(averageOf(runTimes));
    }

    return calculateNodeWeights(averageRunTimes);
}

vector<Measurement> BenchmarkRunner::runBenchmark(const BenchmarkResult& nodeWeights)
{
    return runBenchmark(nodeWeights, *clusterProblem);
}

vector<Measurement> BenchmarkRunner::runBenchmark(const BenchmarkResult& nodeWeights, ProblemStatement& problem)
{
    if (MpiHelper::isMaster())
    {
        scheduler->setNodeset(nodeWeights);
        return benchmarkNodeset(problem);
    }
    else
    {
        getBenchmarked();
        return vector<Measurement>(iterations, Measurement(0));
    }
}

vector<Measurement> BenchmarkRunner::benchmarkNodeset(ProblemStatement& problem)
{
    vector<Measurement> result;

    scheduler->provideData(problem);

    for (size_t i = 0; i < warmUps; ++i)
    {
        measureCall();
    }
    for (size_t i = 0; i < iterations; ++i)
    {
        result.push_back(measureCall());
    }

    scheduler->outputData(problem);

    return result;
}

void BenchmarkRunner::getBenchmarked()
{
    for (size_t i = 0; i < iterations + warmUps; ++i)
        scheduler->dispatch(); // slave side
}

Measurement BenchmarkRunner::measureCall()
{
    high_resolution_clock::time_point before = high_resolution_clock::now();
    scheduler->dispatch();
    return high_resolution_clock::now() - before;
}

BenchmarkResult BenchmarkRunner::calculateNodeWeights(const vector<Measurement>& averageRunTimes)
{
    BenchmarkResult result;
    Measurement totalAverageRunTime(0);

    for (const Measurement& averageRunTime : averageRunTimes)
        totalAverageRunTime += averageRunTime;

    for (size_t i=0; i<averageRunTimes.size(); ++i)
    {
        double average = (double)averageRunTimes[i].count();
        result[i] = 100.0 * average / totalAverageRunTime.count();
    }

    return result;
}

Measurement BenchmarkRunner::averageOf(const vector<Measurement>& runTimes)
{
    if (runTimes.empty())
        return Measurement(0);

    Measurement sum(0);

    for (const Measurement& runTime : runTimes)
        sum += runTime;

    return sum / runTimes.size();
}
