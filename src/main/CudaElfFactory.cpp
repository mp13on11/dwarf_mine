#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
#endif

#include "main/CudaElfFactory.h"

#include <stdexcept>

using namespace std;

unique_ptr<Elf> CudaElfFactory::createElfFrom(const ElfCategory& category) const
{
#ifndef HAVE_CUDA
    throw runtime_error("You have to build with Cuda support in order to create cuda elves of category " + category);
#else
    return unique_ptr<Elf>(new CudaMatrixElf());
#endif
}
