#include "SimpleBenchmarkRunner.h"
#include "common/Configuration.h"
#include "common/ProblemStatement.h"
#include "common/Scheduler.h"
#include "common/SchedulerFactory.h"

using namespace std;
using namespace std::chrono;

typedef SimpleBenchmarkRunner::Measurement Measurement;

SimpleBenchmarkRunner::SimpleBenchmarkRunner(const Configuration& config) :
    warmUps(config.warmUps()), iterations(config.iterations()),
    problemStatement(config.createProblemStatement())
{

    unique_ptr<SchedulerFactory> factory = config.createSchedulerFactory();
    scheduler = factory->createScheduler();
    scheduler->setNodeset(0);
}

vector<Measurement> SimpleBenchmarkRunner::run() const
{
    provideData();
    warmUp();
    vector<Measurement> result = iterate();
    outputData();

    return result;
}

void SimpleBenchmarkRunner::provideData() const
{
    scheduler->provideData(problemStatement->getInput());
}

void SimpleBenchmarkRunner::warmUp() const
{
    for (size_t i=0; i<warmUps; ++i)
        scheduler->dispatchSimple();
}

vector<Measurement> SimpleBenchmarkRunner::iterate() const
{
    vector<Measurement> result;

    for (size_t i=0; i<iterations; ++i)
    {
        startMeasurement();
        scheduler->dispatchSimple();
        result.push_back(endMeasurement());
    }

    return result;
}

void SimpleBenchmarkRunner::outputData() const
{
    scheduler->outputData(problemStatement->getOutput());
}

void SimpleBenchmarkRunner::startMeasurement() const
{
    startOfMeasurement = high_resolution_clock::now();
}

Measurement SimpleBenchmarkRunner::endMeasurement() const
{
    return high_resolution_clock::now() - startOfMeasurement;
}
