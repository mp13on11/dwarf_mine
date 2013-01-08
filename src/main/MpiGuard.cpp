#include "MpiGuard.h"
#include <mpi.h>

const int MpiGuard::MASTER_RANK = 0;

bool MpiGuard::isMaster()
{
    return MPI::COMM_WORLD.Get_rank() == MASTER_RANK;
}

size_t MpiGuard::numberOfNodes()
{
    return MPI::COMM_WORLD.Get_size();
}

MpiGuard::MpiGuard(int argc, char** argv)
{
    MPI::Init(argc, argv);
}

MpiGuard::~MpiGuard()
{
    MPI::Finalize();
}
