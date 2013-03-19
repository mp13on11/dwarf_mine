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
            BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(i); };

            initializeMaster(*generatedProblem, {{i, 1}});
            run(targetMethod);
            finalizeMaster(*generatedProblem);
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
        initializeMaster(*fileProblem, nodeWeights);
        run(targetMethod);
        finalizeMaster(*fileProblem);
    }
    else
    {
        run(targetMethod);
    }
}

void BenchmarkRunner::runElf() const
{
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchSimple(); };

    initializeMaster(*fileProblem);
    run(targetMethod);
    finalizeMaster(*fileProblem);
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

void BenchmarkRunner::initializeMaster(const ProblemStatement& problem, const BenchmarkResult& nodeWeights) const
{
    scheduler->setNodeset(nodeWeights);
    scheduler->provideData(problem);
}

void BenchmarkRunner::finalizeMaster(const ProblemStatement& problem) const
{
    scheduler->outputData(problem);
}
