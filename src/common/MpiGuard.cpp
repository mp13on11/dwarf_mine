#include "MpiGuard.h"
#include <mpi.h>

int MpiGuard::threadSupport;

MpiGuard::MpiGuard(int argc, char** argv)
{
    threadSupport = MPI::Init_thread(argc, argv, MPI_THREAD_MULTIPLE);
}

MpiGuard::~MpiGuard()
{
    MPI::Finalize();
}

int MpiGuard::getThreadSupport()
{
    return threadSupport;
}
