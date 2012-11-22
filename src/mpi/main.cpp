#include "benchmark/BenchmarkKernel.h"
#include "mpi/MpiMatrixKernel.h"

using namespace std;

shared_ptr<BenchmarkKernel> createKernel()
{
    return shared_ptr<BenchmarkKernel>(new MpiMatrixKernel());
}
