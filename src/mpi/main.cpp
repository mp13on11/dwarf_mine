#include "benchmark/BenchmarkKernel.h"
#include "mpi/MpiMatrixKernel.h"
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
    return shared_ptr<BenchmarkKernel>(new MpiMatrixKernel());
}

MpiInitializer::MpiInitializer()
{
    MPI::Init();
}

MpiInitializer::~MpiInitializer()
{
    MPI::Finalize();
}
