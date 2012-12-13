#include "elves/matrix/cuda/CudaMatrixElf.h"
#include "main/CudaElfFactory.h"

using namespace std;

unique_ptr<Elf> CudaElfCategory::createElfFrom(const ElfCategory& category) const
{
    return unique_ptr<Elf>(new CudaMatrixElf());
}
