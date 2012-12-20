#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
#endif

#include "main/CudaElfFactory.h"
#include "matrix/MatrixScheduler.h"

#include <stdexcept>

using namespace std;

CudaElfFactory::CudaElfFactory(const ElfCategory& category)
    : ElfFactory(category)
{

}

unique_ptr<Elf> CudaElfFactory::createElfImplementation() const
{
#ifndef HAVE_CUDA
    throw runtime_error("You have to build with Cuda support in order to create cuda elves of category " + category);
#else
    return unique_ptr<Elf>(new CudaMatrixElf());
#endif
}

unique_ptr<Scheduler> CudaElfFactory::createSchedulerImplementation() const
{
    return unique_ptr<Scheduler>(new MatrixScheduler());
}
