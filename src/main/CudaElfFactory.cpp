#include "matrix/cuda/CudaMatrixElf.h"
#include "main/CudaElfFactory.h"

using namespace std;

unique_ptr<Elf> CudaElfFactory::createElfFrom(const ElfCategory& category) const
{
    return unique_ptr<Elf>(new CudaMatrixElf());
}
