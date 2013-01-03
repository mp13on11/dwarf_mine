#include "Scheduler.h"
#include "ProblemStatement.h"

#include <mpi.h>
#include <stdexcept>

Scheduler::Scheduler() :
    nodesHaveRatings(false), rank(MPI::COMM_WORLD.Get_rank()), elf(nullptr)
{
}

Scheduler::Scheduler(const BenchmarkResult& benchmarkResult) :
    nodesHaveRatings(true), nodeSet(benchmarkResult), rank(MPI::COMM_WORLD.Get_rank()), elf(nullptr)
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
    if (!hasData())
    {
        throw std::runtime_error("Scheduler::dispatch(): No ProblemStatement configured!");
    }

    if (elf == nullptr)
    {
        throw std::runtime_error("Scheduler::dispatch(): No elf configured!");
    }

    if (nodeSet.empty() && rank == MASTER)
    {
        throw std::runtime_error("Scheduler::dispatch(): Nodeset is empty!");
    }

    doDispatch();
}

void Scheduler::dispatch(ProblemStatement& statement)
{
    provideData(statement);
    dispatch();
    outputData(statement);
}
