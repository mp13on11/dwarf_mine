#include "ProblemStatement.h"
#include "Scheduler.h"

#include <stdexcept>

Scheduler::Scheduler() :
    rank(MpiHelper::rank())
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

void Scheduler::dispatch(ProblemStatement& statement)
{
    provideData(statement);
    dispatch();
    outputData(statement);
}
