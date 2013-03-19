#include "BenchmarkRunner.h"
#include "MpiHelper.h"
#include "Profiler.h"
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

void BenchmarkRunner::benchmarkIndividualNodes(Profiler& profiler) const
{
    if (MpiHelper::isMaster())
    {
        for (size_t i=0; i<MpiHelper::numberOfNodes(); ++i)
        {
            BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(i); };

            initializeMaster(*generatedProblem, {{i, 1}});
            run(targetMethod, profiler);
            finalizeMaster(*generatedProblem);
        }
    }
    else
    {
        BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(MpiHelper::rank()); };
        run(targetMethod, profiler);
    }
}

void BenchmarkRunner::runBenchmark(const BenchmarkResult& nodeWeights, Profiler& profiler) const
{
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatch(); };

    if (MpiHelper::isMaster())
    {
        initializeMaster(*fileProblem, nodeWeights);
        run(targetMethod, profiler);
        finalizeMaster(*fileProblem);
    }
    else
    {
        run(targetMethod, profiler);
    }
}

void BenchmarkRunner::runElf(Profiler& profiler) const
{
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchSimple(); };

    initializeMaster(*fileProblem);
    run(targetMethod, profiler);
    finalizeMaster(*fileProblem);
}

void BenchmarkRunner::run(BenchmarkMethod targetMethod, Profiler& profiler) const
{
    for (size_t i = 0; i < warmUps; ++i)
    {
        targetMethod();
    }

    profiler.beginIterationBlock();

    for (size_t i = 0; i < iterations; ++i)
    {
        profiler.beginIteration();
        targetMethod();
        profiler.endIteration();
    }

    profiler.endIterationBlock();
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
