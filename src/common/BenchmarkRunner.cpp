#include "BenchmarkRunner.h"
#include "Communicator.h"
#include "Configuration.h"
#include "Profiler.h"
#include "Scheduler.h"

using namespace std;

BenchmarkRunner::BenchmarkRunner(const Configuration& config) :
        config(&config),
        iterations(config.iterations()), warmUps(config.warmUps()),
        fileProblem(config.createProblemStatement()),
        generatedProblem(config.createGeneratedProblemStatement())
{
}

void BenchmarkRunner::runBenchmark(const Communicator& communicator, Profiler& profiler) const
{
    if (!communicator.isValid())
        return;
    
    unique_ptr<Scheduler> scheduler = config->createScheduler(communicator);

    if (communicator.isMaster())
    {
        scheduler->provideData(*fileProblem);
        run(*scheduler, profiler);
        scheduler->outputData(*fileProblem);
    }
    else
    {
        run(*scheduler, profiler);
    }
}

void BenchmarkRunner::run(Scheduler& scheduler, Profiler& profiler) const
{
    for (size_t i = 0; i < warmUps; ++i)
    {
        scheduler.dispatch();
    }

    profiler.beginIterationBlock();

    for (size_t i = 0; i < iterations; ++i)
    {
        profiler.beginIteration();
        scheduler.dispatch();
        profiler.endIteration();
    }

    profiler.endIterationBlock();
}
