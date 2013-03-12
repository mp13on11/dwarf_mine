#include "ProblemStatement.h"
#include "Scheduler.h"

#include <stdexcept>

Scheduler::Scheduler()
{
}

Scheduler::~Scheduler()
{
}

void Scheduler::setNodeset(const BenchmarkResult& benchmarkResult)
{
    nodeSet = benchmarkResult;
}

void Scheduler::setNodeset(NodeId singleNode)
{
    nodeSet = {{singleNode, 0}};
}

void Scheduler::provideData(const ProblemStatement& problem)
{
	if (problem.hasInput())
	{
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
