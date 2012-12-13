#include "main/ElfFactory.h"

#include <stdexcept>

using namespace std;

unique_ptr<Elf> ElfFactory::createElf(const ElfCategory& category) const
{
    validate(category);

    return createElfFrom(category);
}

void ElfFactory::validate(const ElfCategory& category) const
{
    if (category != "matrix")
        throw runtime_error("Unknown elf category: " + category);
}
