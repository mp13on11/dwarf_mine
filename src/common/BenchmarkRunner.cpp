#include "BenchmarkRunner.h"
#include "MpiHelper.h"
#include "SchedulerFactory.h"

using namespace std;

BenchmarkRunner::BenchmarkRunner(Configuration& config) :
        config(&config),
        iterations(config.iterations()), warmUps(config.warmUps()),
        fileProblem(config.createProblemStatement()),
        generatedProblem(config.createGeneratedProblemStatement()),
        scheduler(config.createScheduler())
{
}

void BenchmarkRunner::benchmarkIndividualNodes() const
{
    if (MpiHelper::isMaster())
    {
        for (size_t i=0; i<MpiHelper::numberOfNodes(); ++i)
        {
            scheduler->setNodeset({{i, 1}});
            BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(i); };
            benchmarkNodeset(*generatedProblem, targetMethod);
        }
    }
    else
    {
        BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(MpiHelper::rank()); };
        run(targetMethod);
    }
}

void BenchmarkRunner::runBenchmark(const BenchmarkResult& nodeWeights) const
{
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatch(); };

    if (MpiHelper::isMaster())
    {
        scheduler->setNodeset(nodeWeights);
        benchmarkNodeset(*fileProblem, targetMethod);
    }
    else
    {
        run(targetMethod);
    }
}

void BenchmarkRunner::runElf() const
{
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchSimple(); };
    scheduler->setNodeset({{0, 0}});
    benchmarkNodeset(*fileProblem, targetMethod);
}

void BenchmarkRunner::benchmarkNodeset(const ProblemStatement& problem, BenchmarkMethod targetMethod) const
{
    scheduler->provideData(problem);
    run(targetMethod);
    scheduler->outputData(problem);
}

void BenchmarkRunner::run(BenchmarkMethod targetMethod) const
{
    for (size_t i = 0; i < warmUps; ++i)
    {
        targetMethod();
    }
    for (size_t i = 0; i < iterations; ++i)
    {
        targetMethod();
    }
}
