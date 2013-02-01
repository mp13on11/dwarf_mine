#include "SimpleFactorizationScheduler.h"
#include "SimpleMatrixScheduler.h"
#include "SimpleSchedulerFactory.h"
#include "factorization_montecarlo/MonteCarloFactorizationElf.h"
#include "matrix/smp/SMPMatrixElf.h"

#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
#endif

using namespace std;

unique_ptr<SchedulerFactory> SimpleSchedulerFactory::createFor(const string& type, const ElfCategory& category)
{
    validateType(type);
    validateCategory(category);

    return unique_ptr<SchedulerFactory>(
            new SimpleSchedulerFactory(createFactory(type, category))
        );
}

SimpleSchedulerFactory::SimpleSchedulerFactory(const function<Scheduler*()>& factory) :
        SchedulerFactory(factory)
{
}

function<Scheduler*()> SimpleSchedulerFactory::createFactory(const string& type, const ElfCategory& category)
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

function<Scheduler*()> SimpleSchedulerFactory::createSmpFactory(const ElfCategory& category)
{
    if (category == "matrix")
        return SchedulerFactory::createFactory<SimpleMatrixScheduler, SMPMatrixElf>();
    else
        return SchedulerFactory::createFactory<SimpleFactorizationScheduler, MonteCarloFactorizationElf>();
}

#ifdef HAVE_CUDA
function<Scheduler*()> SimpleSchedulerFactory::createCudaFactory(const ElfCategory& category)
{
    if (category == "matrix")
        return SchedulerFactory::createFactory<SimpleMatrixScheduler, CudaMatrixElf>();
    else
        throw runtime_error(
            "No CUDA elf implemented for Monte Carlo Factorization"
        );
}
#endif
