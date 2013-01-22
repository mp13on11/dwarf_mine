#include "SchedulerFactory.h"
#include "factorize/FactorizationScheduler.h"
#include "factorize/smp/SmpFactorizationElf.h"
#include "matrix/MatrixScheduler.h"
#include "matrix/smp/SMPMatrixElf.h"

#ifdef HAVE_CUDA
#include "factorize/cuda/CudaFactorizationElf.h"
#include "matrix/cuda/CudaMatrixElf.h"
#endif

#include <stdexcept>

using namespace std;

unique_ptr<SchedulerFactory> SchedulerFactory::createFor(const string& type, const ElfCategory& category)
{
    validateType(type);
    validateCategory(category);

    return unique_ptr<SchedulerFactory>(
            new SchedulerFactory(createFactory(type, category))
        );
}

SchedulerFactory::SchedulerFactory(const function<Scheduler*()>& factory) :
        factory(factory)
{
}

SchedulerFactory::~SchedulerFactory()
{
}

unique_ptr<Scheduler> SchedulerFactory::createScheduler() const
{
    return unique_ptr<Scheduler>(factory());
}

void SchedulerFactory::validateType(const string& type)
{
    if (type != "cuda" && type != "smp")
        throw runtime_error("Unknown scheduler type " + type + " in " __FILE__);

#ifndef HAVE_CUDA
    if (type == "cuda")
        throw runtime_error("You have to build with Cuda support in order to create cuda elves in " __FILE__);
#endif
}

void SchedulerFactory::validateCategory(const ElfCategory& category)
{
    if (category != "factorize" && category != "matrix")
        throw runtime_error("Unknown elf category: " + category + " in " __FILE__);
}

function<Scheduler*()> SchedulerFactory::createFactory(const string& type, const ElfCategory& category)
{
    if (type == "smp")
        return createSmpFactory(category);
#ifdef HAVE_CUDA
    else if (type == "cuda")
        return createCudaFactory(category);
#endif
    else
        throw runtime_error(
                "This is here to make the compiler happy in the case"
                " when HAVE_CUDA is not defined..."
            );
}

function<Scheduler*()> SchedulerFactory::createSmpFactory(const ElfCategory& category)
{
    if (category == "matrix")
        return createFactory<MatrixScheduler, SMPMatrixElf>();
    else
        return createFactory<FactorizationScheduler, SmpFactorizationElf>();
}

#ifdef HAVE_CUDA
function<Scheduler*()> SchedulerFactory::createCudaFactory(const ElfCategory& category)
{
    if (category == "matrix")
        return createFactory<MatrixScheduler, CudaMatrixElf>();
    else
        return createFactory<FactorizationScheduler, CudaFactorizationElf>();
}
#endif
