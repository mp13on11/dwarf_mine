#include "MpiGuard.h"

#include <mpi.h>

int MpiGuard::threadSupport;

MpiGuard::MpiGuard(bool multiThreaded, int argc, char** argv)
{
    int requiredThreadSupport = multiThreaded ?
        MPI_THREAD_MULTIPLE :
        MPI_THREAD_SINGLE;
    threadSupport = MPI::Init_thread(argc, argv, requiredThreadSupport);
}

MpiGuard::~MpiGuard()
{
    MPI::Finalize();
}

int MpiGuard::getThreadSupport()
{
    return threadSupport;
}
