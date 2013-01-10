#include "ProblemStatement.h"
#include "Scheduler.h"

#include <stdexcept>

Scheduler::Scheduler() :
    rank(MpiHelper::rank()), elf(nullptr)
{
}

Scheduler::Scheduler(const BenchmarkResult& benchmarkResult) :
    nodeSet(benchmarkResult), rank(MpiHelper::rank()), elf(nullptr)
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

void Scheduler::dispatch()
{
    if (elf == nullptr)
    {
        throw std::runtime_error("Scheduler::dispatch(): No elf configured!");
    }

    if (MpiHelper::isMaster(rank))
    {
        if (!hasData())
        {
            throw std::runtime_error("Scheduler::dispatch(): No ProblemStatement configured!");
        }

        if (nodeSet.empty())
        {
            throw std::runtime_error("Scheduler::dispatch(): Nodeset is empty!");
        }
    }
    doDispatch();
}

void Scheduler::dispatch(ProblemStatement& statement)
{
    provideData(statement);
    dispatch();
    outputData(statement);
}
