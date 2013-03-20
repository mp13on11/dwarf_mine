#include "BenchmarkRunner.h"
#include "Communicator.h"
#include "Profiler.h"
#include "Scheduler.h"
#include "SchedulerFactory.h"

using namespace std;

BenchmarkRunner::BenchmarkRunner(const Configuration& config) :
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

    unique_ptr<Scheduler> scheduler = config->createScheduler(communicator);
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchBenchmark(communicator.size()-1); };

    if (communicator.isMaster())
    {
        scheduler->provideData(*generatedProblem);

        run(targetMethod, profiler);
        
        scheduler->outputData(*generatedProblem);
    }
    else
    {
        run(targetMethod, profiler);
    }
}

void BenchmarkRunner::runBenchmark(const Communicator& communicator, Profiler& profiler) const
{
    if (!communicator.isValid())
        return;
    
    unique_ptr<Scheduler> scheduler = config->createScheduler(communicator);
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatch(); };

    if (communicator.isMaster())
    {
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
    
    unique_ptr<Scheduler> scheduler = config->createScheduler(communicator);
    BenchmarkMethod targetMethod = [&](){ scheduler->dispatchSimple(); };

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
