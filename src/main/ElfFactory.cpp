#include "main/ElfFactory.h"

#include <stdexcept>

using namespace std;

ElfFactory::ElfFactory(const ElfCategory& category)
    : _category(category)
{
}

unique_ptr<Elf> ElfFactory::createElf() const
{
    validate();

    return createElfImplementation();
}

unique_ptr<Scheduler> ElfFactory::createScheduler() const
{
    validate();

    return createSchedulerImplementation();
}

void ElfFactory::validate() const
{
    if (_category != "matrix")
        throw runtime_error("Unknown elf category: " + _category);
}
