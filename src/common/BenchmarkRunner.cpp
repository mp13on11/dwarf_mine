#include "BenchmarkRunner.h"
#include "Communicator.h"
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

void BenchmarkRunner::benchmarkNode(const Communicator& communicator, Profiler& profiler) const
{
    if (!communicator.isValid())
        return;

    unique_ptr<Scheduler> scheduler = config->createScheduler();
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(Communicator::MASTER_RANK); };

    if (MpiHelper::isMaster())
    {
        scheduler->setNodeset({{Communicator::MASTER_RANK, 1}});
        scheduler->provideData(*generatedProblem);

        run(targetMethod, profiler);
        
        scheduler->outputData(*generatedProblem);
    }
    else if (MpiHelper::rank() == -1)
    {
        run(targetMethod, profiler);
    }
}

void BenchmarkRunner::runBenchmark(const Communicator& communicator, Profiler& profiler) const
{
    if (!communicator.isValid())
        return;
    
    unique_ptr<Scheduler> scheduler = config->createScheduler();
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatch(); };

    if (MpiHelper::isMaster())
    {
        BenchmarkResult nodeSet;
        for (size_t i=0; i<communicator.size(); ++i)
        {
            nodeSet[i] = communicator.weights()[i];
        }
        scheduler->setNodeset(nodeSet);
        scheduler->provideData(*fileProblem);

        run(targetMethod, profiler);
        
        scheduler->outputData(*fileProblem);
    }
    else
    {
        run(targetMethod, profiler);
    }
}

void BenchmarkRunner::runElf(const Communicator& communicator, Profiler& profiler) const
{
    if (!communicator.isValid())
        return;
    
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
