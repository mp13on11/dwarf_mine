#include "MpiGuard.h"
#include <mpi.h>

MpiGuard::MpiGuard(int argc, char** argv)
{
    MPI::Init(argc, argv);
}

MpiGuard::~MpiGuard()
{
    MPI::Finalize();
}
