#include "SchedulerFactory.h"
#include "SMPElfFactory.h"
#include "CudaElfFactory.h"

#include <stdexcept>

using namespace std;

SchedulerFactory::SchedulerFactory(const ElfCategory& category)
    : _category(category)
{
}

unique_ptr<Scheduler> SchedulerFactory::createScheduler() const
{
    validate();

    return createSchedulerImplementation();
}

void SchedulerFactory::validate() const
{
    if (_category != "matrix" && _category != "factorize")
        throw runtime_error("Unknown elf category: " + _category + " in " __FILE__);
}

unique_ptr<SchedulerFactory> createElfFactory(const std::string& type, const ElfCategory& category)
{
    if (type == "smp")
        return unique_ptr<SchedulerFactory>(new SMPElfFactory(category));
    else if (type == "cuda")
        return unique_ptr<SchedulerFactory>(new CudaElfFactory(category));
    else
        throw runtime_error("createElfFactory(): type must be one of cuda|smp");

    return nullptr; // avoid warning
}

