#include "BenchmarkRunner.h"
#include "MpiHelper.h"
#include "Profiler.h"
#include "Scheduler.h"
#include "SchedulerFactory.h"

using namespace std;

BenchmarkRunner::BenchmarkRunner(Configuration& config) :
        config(&config),
        iterations(config.iterations()), warmUps(config.warmUps()),
        fileProblem(config.createProblemStatement()),
        generatedProblem(config.createGeneratedProblemStatement())
{
}

void BenchmarkRunner::benchmarkNode(int node, Profiler& profiler) const
{
    unique_ptr<Scheduler> scheduler = config->createScheduler();
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(node); };

    if (MpiHelper::isMaster())
    {
        scheduler->setNodeset({{node, 1}});
        scheduler->provideData(*generatedProblem);

        run(targetMethod, profiler);
        
        scheduler->outputData(*generatedProblem);
    }
    else if (MpiHelper::rank() == node)
    {
        run(targetMethod, profiler);
    }
}

void BenchmarkRunner::runBenchmark(const BenchmarkResult& nodeWeights, Profiler& profiler) const
{
    unique_ptr<Scheduler> scheduler = config->createScheduler();
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatch(); };

    if (MpiHelper::isMaster())
    {
        scheduler->setNodeset(nodeWeights);
        scheduler->provideData(*fileProblem);

        run(targetMethod, profiler);
        
        scheduler->outputData(*fileProblem);
    }
    else
    {
        run(targetMethod, profiler);
    }
}

void BenchmarkRunner::runElf(Profiler& profiler) const
{
    unique_ptr<Scheduler> scheduler = config->createScheduler();
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchSimple(); };

    scheduler->setNodeset({{0, 1}});
    scheduler->provideData(*fileProblem);

    run(targetMethod, profiler);
    
    scheduler->outputData(*fileProblem);
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
