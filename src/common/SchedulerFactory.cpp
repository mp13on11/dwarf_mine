/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#include "SchedulerFactory.h"
#include "quadratic_sieve/QuadraticSieveScheduler.h"
#include "quadratic_sieve/smp/SmpQuadraticSieveElf.h"
#include "factorization_montecarlo/FactorizationScheduler.h"
#include "factorization_montecarlo/MonteCarloFactorizationElf.h"
#include "matrix/MatrixScheduler.h"
#include "matrix/MatrixOnlineScheduler.h"
#include "matrix/smp/SMPMatrixElf.h"
#include "othello_montecarlo/OthelloScheduler.h"
#include "othello_montecarlo/smp/SMPOthelloElf.h"

#ifdef HAVE_CUDA
#include "matrix/cuda/CudaMatrixElf.h"
#include "quadratic_sieve/cuda/CudaQuadraticSieveElf.h"
#include "othello_montecarlo/cuda/CudaOthelloElf.h"
#endif

#include <stdexcept>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/map.hpp>

using namespace std;

#ifndef HAVE_CUDA
struct HasNoCudaDummy : public Elf {};

typedef HasNoCudaDummy CudaMatrixElf, CudaQuadraticSieveElf, CudaOthelloElf;
#endif

typedef SchedulerFactory::FactoryFunction FactoryFunction;

template<typename SchedulerType, typename ElfType>
static FactoryFunction innerCreateFactory()
{
    return [](const Communicator& communicator)
    { 
        return new SchedulerType(communicator, []()
            {
                return new ElfType();
            }
        ); 
    };
}

template<typename SchedulerType, typename SmpElfType, typename CudaElfType>
static FactoryFunction createFactory(bool useCuda)
{
    if (useCuda)
    {
#ifdef HAVE_CUDA
        return innerCreateFactory<SchedulerType, CudaElfType>();
#else
        throw runtime_error("You have to build with Cuda support in order to create cuda elves in " __FILE__);
#endif
    }
    else
    {
        return innerCreateFactory<SchedulerType, SmpElfType>();
    }
}

template<typename SchedulerType, typename SmpElfType>
static FactoryFunction createFactory(bool useCuda)
{
    if (useCuda)
        throw runtime_error("This category has no Cuda implementation in " __FILE__);

    return innerCreateFactory<SchedulerType, SmpElfType>();
}

static map<string, function<FactoryFunction(bool)>> sFactoryFunctionsMap =
{
    {
        "matrix",
        &createFactory<MatrixScheduler, SMPMatrixElf, CudaMatrixElf>
    },
    {
        "matrix_online",
        &createFactory<MatrixOnlineScheduler, SMPMatrixElf, CudaMatrixElf>
    },
    {
        "factorization_montecarlo",
        &createFactory<FactorizationScheduler, MonteCarloFactorizationElf>
    },
    {
        "quadratic_sieve",
        &createFactory<QuadraticSieveScheduler, SmpQuadraticSieveElf, CudaQuadraticSieveElf>
    },
    {
        "montecarlo_tree_search",
        &createFactory<OthelloScheduler, SMPOthelloElf, CudaOthelloElf>
    }
};

static void validateType(const string& type)
{
    if (type != "cuda" && type != "smp")
        throw runtime_error("Unknown scheduler type " + type + " in " __FILE__);
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
    boost::copy(
            sFactoryFunctionsMap | boost::adaptors::map_keys, 
            std::back_inserter(categories)
    );
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

unique_ptr<Scheduler> SchedulerFactory::createScheduler(const Communicator& communicator) const
{
    return unique_ptr<Scheduler>(factory(communicator));
}
