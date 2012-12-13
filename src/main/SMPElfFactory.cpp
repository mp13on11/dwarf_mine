#include "matrix/smp/SMPMatrixElf.h"
#include "main/SMPElfFactory.h"

using namespace std;

unique_ptr<Elf> SMPElfFactory::createElfFrom(const ElfCategory& category) const
{
    return unique_ptr<Elf>(new SMPMatrixElf());
}
