#include "MpiHelper.h"
#include <mpi.h>

const NodeId MpiHelper::MASTER = 0;

bool MpiHelper::isMaster()
{
    return isMaster(rank());
}

bool MpiHelper::isMaster(NodeId id)
{
    return id == MASTER;
}

NodeId MpiHelper::rank()
{
    return MPI::COMM_WORLD.Get_rank();
}

size_t MpiHelper::numberOfNodes()
{
    return MPI::COMM_WORLD.Get_size();
}
