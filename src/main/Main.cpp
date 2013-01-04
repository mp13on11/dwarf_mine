#include <exception>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <functional>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <ctime>
#include <boost/lexical_cast.hpp>

#include "BenchmarkRunner.h"
#include "Configuration.h"
#include "matrix/MatrixHelper.h"
#include "matrix/Matrix.h"

using namespace std;

void generateProblemData(ProblemStatement& statement)
{
    Matrix<float> first(600,416);
    Matrix<float> second(416, 808);
    auto distribution = uniform_real_distribution<float> (-100, +100);
    auto engine = mt19937(time(nullptr));
    auto generator = bind(distribution, engine);
    MatrixHelper::fill(first, generator);
    MatrixHelper::fill(second, generator);
    MatrixHelper::writeMatrixTo(*(statement.input), first);
    MatrixHelper::writeMatrixTo(*(statement.input), second);
}

class MPIGuard
{
public:
    MPIGuard(int argc, char** argv)
    {
        MPI::Init(argc, argv);
    }

    ~MPIGuard()
    {
        MPI::Finalize();
    }
};

int main(int argc, char** argv)
{
    Configuration config(argc, argv);
    // used to ensure MPI::Finalize is called on exit of the application
    auto mpiGuard = MPIGuard(argc, argv);
    try
    {
        ProblemStatement benchmarkStatement("matrix");
        generateProblemData(benchmarkStatement);
        unique_ptr<ElfFactory> factory(config.getElfFactory(benchmarkStatement.elfCategory));
        BenchmarkRunner runner(50);

        runner.runBenchmark(benchmarkStatement, *factory);
/*
        auto results = runner.getResults();

        auto statement = config.createProblemStatement("matrix");
        auto scheduler = factory->createScheduler();
        scheduler->setNodeset(results);
        auto elf = factory->createElf();
        scheduler->setElf(elf.get());

        scheduler->dispatch(*statement);
        */

    }
    catch (exception &e)
    {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "FATAL ERROR of unknown type." << endl;
        return 1;
    }

    return 0;
}

