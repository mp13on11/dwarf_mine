#include "main/ElfFactory.h"
#include "main/SMPElfFactory.h"
#include "main/CudaElfFactory.h"

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
    if (_category != "matrix" && _category != "factorize")
        throw runtime_error("Unknown elf category: " + _category + " in " __FILE__);
}

unique_ptr<ElfFactory> createElfFactory(const std::string& type, const ElfCategory& category)
{
    if (type == "smp")
        return unique_ptr<ElfFactory>(new SMPElfFactory(category));
    else if (type == "cuda")
        return unique_ptr<ElfFactory>(new CudaElfFactory(category));
    else
        throw runtime_error("createElfFactory(): type must be one of cuda|smp");

    return nullptr; // avoid warning
}

