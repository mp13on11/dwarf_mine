#pragma once

#include <mpi.h>

const int MASTER = 0;

typedef int DeviceId;
typedef int NodeId;

class MPIGuard
{
public:
    MPIGuard(int argc, char** argv)
    {
        MPI::Init(argc, argv);
    }

    static bool isMaster()
    {
        return MPI::COMM_WORLD.Get_rank() == MASTER;
    }

    static size_t numberOfNodes()
    {
        return (size_t)MPI::COMM_WORLD.Get_size();
    }

    ~MPIGuard()
    {
        MPI::Finalize();
    }
};