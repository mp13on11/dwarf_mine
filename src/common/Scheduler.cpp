#include "ProblemStatement.h"
#include "Scheduler.h"

#include <stdexcept>

Scheduler::Scheduler()
{
}

Scheduler::~Scheduler()
{
}

void Scheduler::provideData(const ProblemStatement& problem)
{
    if (problem.hasInput())
    {
        problem.getInput().clear();
        problem.getInput().seekg(0, std::ios::beg);
        provideData(problem.getInput());
    }
    else
    {
        generateData(problem.getDataGenerationParameters());
    }
}

void Scheduler::outputData(const ProblemStatement& problem)
{
	outputData(problem.getOutput());
}
