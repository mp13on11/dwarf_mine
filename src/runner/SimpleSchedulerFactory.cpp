#include "SimpleFactorizationScheduler.h"
#include "SimpleMatrixScheduler.h"
#include "SimpleSchedulerFactory.h"
#include "SimpleMonteCarloScheduler.h"
#include "factorize/smp/SmpFactorizationElf.h"
#include "montecarlo/smp/SMPMonteCarloElf.h"
#include "matrix/smp/SMPMatrixElf.h"

#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
#include "factorize/cuda/CudaFactorizationElf.h"
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
    else if (category == "montecarlo")
        return SchedulerFactory::createFactory<SimpleMonteCarloScheduler, SMPMonteCarloElf>();
    else
        return SchedulerFactory::createFactory<SimpleFactorizationScheduler, SmpFactorizationElf>();
}

#ifdef HAVE_CUDA
function<Scheduler*()> SimpleSchedulerFactory::createCudaFactory(const ElfCategory& category)
{
    if (category == "matrix")
        return SchedulerFactory::createFactory<SimpleMatrixScheduler, CudaMatrixElf>();
    else if (category == "montecarlo")
        throw runtime_error("CUDA for MonteCarlo not yet implemnted");
    else
        return SchedulerFactory::createFactory<SimpleFactorizationScheduler, CudaFactorizationElf>();
}
#endif
