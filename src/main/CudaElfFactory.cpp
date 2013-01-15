#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
#include "factorize/cuda/CudaFactorizationElf.h"
#endif

#include "main/CudaElfFactory.h"
#include "matrix/MatrixScheduler.h"
#include "factorize/FactorizationScheduler.h"

#include <stdexcept>

using namespace std;

CudaElfFactory::CudaElfFactory(const ElfCategory& category) :
    ElfFactory(category)
{
}

unique_ptr<Scheduler> CudaElfFactory::createSchedulerImplementation() const
{
#ifndef HAVE_CUDA
	throw runtime_error("You have to build with Cuda support in order to create cuda elves of category " + _category);
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
