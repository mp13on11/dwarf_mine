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
