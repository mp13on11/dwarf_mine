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

SchedulerFactory::SchedulerFactory(const string& type, const ElfCategory& category)
    : _type(type), _category(category)
{
	if (type != "cuda" && type != "smp")
		throw runtime_error("Unknown scheduler type " + type + " in " __FILE__);

#ifndef HAVE_CUDA
	if (type == "cuda")
	    throw runtime_error("You have to build with Cuda support in order to create cuda elves");
#endif

    if (_category != "factorize" && _category != "matrix")
        throw runtime_error("Unknown elf category: " + _category + " in " __FILE__);
}

SchedulerFactory::~SchedulerFactory()
{
}

unique_ptr<Scheduler> SchedulerFactory::createScheduler() const
{
	if (_type == "cuda")
		return createCudaScheduler();
	else
		return createSmpScheduler();
}

unique_ptr<Scheduler> SchedulerFactory::createCudaScheduler() const
{
#ifndef HAVE_CUDA
	return nullptr;
#else
    if (_category == "matrix")
    {
        return unique_ptr<Scheduler>(
                new MatrixScheduler(
                        []() { return new CudaMatrixElf(); }
                    )
            );
    }
    else
    {
        return unique_ptr<Scheduler>(
                new FactorizationScheduler(
                        []() { return new CudaFactorizationElf(); }
                    )
            );
    }
#endif
}

unique_ptr<Scheduler> SchedulerFactory::createSmpScheduler() const
{
    if (_category == "matrix")
    {
        return unique_ptr<Scheduler>(
                new MatrixScheduler(
                        []() { return new SMPMatrixElf(); }
                    )
            );
    }
    else
    {
        return unique_ptr<Scheduler>(
                new FactorizationScheduler(
                        []() { return new SmpFactorizationElf(); }
                    )
            );
    }
}
