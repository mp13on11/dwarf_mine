#include "MpiGuard.h"
#include "Configuration.h"

#include <mpi.h>

int MpiGuard::threadSupport;

MpiGuard::MpiGuard(const Configuration& configuration, int argc, char** argv)
{
    int requiredThreadSupport = configuration.mpiThreadMultiple() ?
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
