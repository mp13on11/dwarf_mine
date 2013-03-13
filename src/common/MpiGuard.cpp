#include "MpiGuard.h"
#include <mpi.h>

MpiGuard::MpiGuard(int argc, char** argv)
{
    MPI::Init_thread(argc, argv, MPI_THREAD_MULTIPLE);
}

MpiGuard::~MpiGuard()
{
    MPI::Finalize();
}
