#include "benchmark/BenchmarkKernel.h"
#include "mpi/MpiMatrixKernel.h"
#include "mpi/MpiStarterKernel.h"

#include <mpi.h>

using namespace std;

class MpiInitializer
{
public:
    MpiInitializer();
    ~MpiInitializer();
} init;

shared_ptr<BenchmarkKernel> createKernel()
{
    shared_ptr<BenchmarkKernel> kernel(new MpiMatrixKernel());

    if (MpiStarterKernel::wasStartedCorrectly())
        return kernel;
    else
        return shared_ptr<BenchmarkKernel>(new MpiStarterKernel(kernel));
}

MpiInitializer::MpiInitializer()
{
    MPI::Init();
}

MpiInitializer::~MpiInitializer()
{
    MPI::Finalize();
}
