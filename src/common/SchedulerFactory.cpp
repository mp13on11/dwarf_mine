#include "SchedulerFactory.h"
#include "quadratic_sieve/QuadraticSieveScheduler.h"
#include "quadratic_sieve/smp/SmpQuadraticSieveElf.h"
#include "factorization_montecarlo/FactorizationScheduler.h"
#include "factorization_montecarlo/MonteCarloFactorizationElf.h"
#include "matrix/MatrixScheduler.h"
#include "matrix/smp/SMPMatrixElf.h"

#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
#include "quadratic_sieve/cuda/CudaQuadraticSieveElf.h"
#endif

#include <stdexcept>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/map.hpp>

using namespace std;

typedef SchedulerFactory::FactoryFunction FactoryFunction;

template<typename SchedulerType, typename ElfType>
static FactoryFunction createFactory()
{
    return []()
        {
            return new SchedulerType([]() { return new ElfType(); });
        };
}

static FactoryFunction createMatrixFactory(bool useCuda)
{
    if (useCuda)
        return createFactory<MatrixScheduler, CudaMatrixElf>();
    else
        return createFactory<MatrixScheduler, SMPMatrixElf>();
}

static FactoryFunction createMonteCarloFactorizationFactory(bool useCuda)
{
    if (useCuda)
        throw runtime_error(
            "No CUDA elf implemented for Monte Carlo Factorization"
        );
    else
        return createFactory<FactorizationScheduler, MonteCarloFactorizationElf>();
}

static FactoryFunction createQuadraticSieveFactory(bool useCuda)
{
    if (useCuda)
        return createFactory<QuadraticSieveScheduler, CudaQuadraticSieveElf>();
    else
        return createFactory<QuadraticSieveScheduler, SmpQuadraticSieveElf>();
}

static map<string, function<FactoryFunction(bool)>> sFactoryFunctionsMap =
{
    { "matrix", createMatrixFactory },
    { "factorization_montecarlo", createMonteCarloFactorizationFactory },
    { "quadratic_sieve", createQuadraticSieveFactory }
};

static void validateType(const string& type)
{
    if (type != "cuda" && type != "smp")
        throw runtime_error("Unknown scheduler type " + type + " in " __FILE__);

#ifndef HAVE_CUDA
    if (type == "cuda")
        throw runtime_error("You have to build with Cuda support in order to create cuda elves in " __FILE__);
#endif
}

static FactoryFunction createFactory(const string& type, const ElfCategory& category)
{
    validateType(type);

    auto factoryCreatorIt = sFactoryFunctionsMap.find(category);

    if (factoryCreatorIt == sFactoryFunctionsMap.end())
        throw runtime_error("Unknown elf category: " + category + " in " __FILE__);

    return factoryCreatorIt->second(type == "cuda");
}

vector<string> SchedulerFactory::getValidCategories()
{
    vector<string> categories;
    boost::copy(sFactoryFunctionsMap | boost::adaptors::map_keys, std::back_inserter(categories));
    return categories;
}

unique_ptr<SchedulerFactory> SchedulerFactory::createFor(const string& type, const ElfCategory& category)
{
    return unique_ptr<SchedulerFactory>(
        new SchedulerFactory(createFactory(type, category))
    );
}

SchedulerFactory::SchedulerFactory(const FactoryFunction& factory) :
    factory(factory)
{
}

unique_ptr<Scheduler> SchedulerFactory::createScheduler() const
{
    return unique_ptr<Scheduler>(factory());
}
