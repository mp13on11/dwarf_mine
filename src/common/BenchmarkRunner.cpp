#include "BenchmarkRunner.h"
#include "MpiHelper.h"
#include "SchedulerFactory.h"

using namespace std;
using namespace std::chrono;

typedef BenchmarkRunner::Measurement Measurement;

/**
 * BenchmarkRunner determines the available devices and benchmarks them idenpendently
 */
BenchmarkRunner::BenchmarkRunner(Configuration& config) :
        config(&config),
        iterations(config.iterations()), warmUps(config.warmUps()),
        clusterProblem(config.createProblemStatement()),
        scheduler(config.createScheduler())
{
}

BenchmarkResult BenchmarkRunner::benchmarkIndividualNodes() const
{
    vector<Measurement> averageRunTimes;
    inPreBenchmark = true;

    for (size_t i=0; i<MpiHelper::numberOfNodes(); ++i)
    {
        vector<Measurement> runTimes = runBenchmark(
            {{static_cast<NodeId>(i), 1}}, false
        );
        averageRunTimes.push_back(averageOf(runTimes));
    }

    return calculateNodeWeights(averageRunTimes);
}

vector<Measurement> BenchmarkRunner::runBenchmark(const BenchmarkResult& nodeWeights) const
{
    inPreBenchmark = false;
    return runBenchmark(nodeWeights, true);
}

vector<Measurement> BenchmarkRunner::runBenchmark(const BenchmarkResult& nodeWeights, bool useProblemStatement) const
{
    BenchmarkMethod targetMethod;

    if (inPreBenchmark)
    {
        targetMethod = [&]()
            {
                scheduler->dispatchBenchmark(nodeWeights.begin()->first);
            };
    }
    else
    {
        targetMethod = [&]()
            {
                scheduler->dispatch();
            };
    }

    if (MpiHelper::isMaster())
    {
        scheduler->setNodeset(nodeWeights);
        scheduler->configureWith(*config);
        return benchmarkNodeset(useProblemStatement, targetMethod);
    }
    else if (slaveShouldRunWith(nodeWeights))
    {
        benchmarkSlave(targetMethod);
    }

    return vector<Measurement>(iterations, Measurement(0));
}

vector<Measurement> BenchmarkRunner::benchmarkNodeset(bool useProblemStatement, BenchmarkMethod targetMethod) const
{
    vector<Measurement> result;

    if (useProblemStatement && clusterProblem->hasInput())
    {
        scheduler->provideData(clusterProblem->getInput());
    }
    else
    {
        scheduler->generateData(clusterProblem->getDataGenerationParameters());
    }

    for (size_t i = 0; i < warmUps; ++i)
    {
        measureCall(targetMethod);
    }
    for (size_t i = 0; i < iterations; ++i)
    {
        result.push_back(measureCall(targetMethod));
    }

    if (useProblemStatement)
        scheduler->outputData(clusterProblem->getOutput());

    return result;
}

void BenchmarkRunner::benchmarkSlave(BenchmarkMethod targetMethod) const
{
    for (size_t i = 0; i < iterations + warmUps; ++i)
    {
        targetMethod();
    }
}

Measurement BenchmarkRunner::measureCall(BenchmarkMethod targetMethod) const
{
    auto before = high_resolution_clock::now();
    targetMethod();
    return duration_cast<Measurement>(high_resolution_clock::now() - before);
}

BenchmarkResult BenchmarkRunner::calculateNodeWeights(const vector<Measurement>& averageRunTimes)
{
    BenchmarkResult result;
    double totalPerformance = 0;

    for (const Measurement& averageRunTime : averageRunTimes)
        totalPerformance += 1.0 / averageRunTime.count();

    for (size_t i=0; i<averageRunTimes.size(); ++i)
    {
        double performance = 1.0 / averageRunTimes[i].count();
        result[i] = performance / totalPerformance;
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

bool BenchmarkRunner::slaveShouldRunWith(const BenchmarkResult& nodeWeights)
{
    // returns true if the slave's rank is included in the nodeWeights
    return nodeWeights.find(MpiHelper::rank()) != nodeWeights.end();
}
