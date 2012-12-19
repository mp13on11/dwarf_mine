#include "Scheduler.h"
#include "ProblemStatement.h"

#include <mpi.h>

Scheduler::Scheduler() :
    nodesHaveRatings(false), rank(MPI::COMM_WORLD.Get_rank())
{  
}

Scheduler::Scheduler(const BenchmarkResult& benchmarkResult) : 
    nodesHaveRatings(true), nodeSet(benchmarkResult), rank(MPI::COMM_WORLD.Get_rank())
{
}

void Scheduler::setNodeset(const BenchmarkResult& benchmarkResult) {
    nodeSet = benchmarkResult;
}

void Scheduler::setNodeset(NodeId singleNode) {
    nodeSet = {{singleNode, 0}};
}
