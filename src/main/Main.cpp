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

#include "BenchmarkRunner.h"
#include "Configuration.h"
#include "matrix/MatrixHelper.h"
#include "matrix/Matrix.h"

using namespace std;

void generateProblemData(ProblemStatement& statement)
{
    Matrix<float> first(100,100);
    Matrix<float> second(100, 100);
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
        BenchmarkRunner runner(100);

        runner.runBenchmark(benchmarkStatement, *factory);

        auto results = runner.getResults();
        for (const auto& result: results)
        {
            cout << result.first << " - " <<result.second<<endl;
        }

        auto statement = config.createProblemStatement("matrix");
        //ProblemStatement statement("matrix");
        //generateProblemData(statement);
        
        auto scheduler = factory->createScheduler();
        scheduler->setNodeset(results);
        auto elf = factory->createElf();
        scheduler->setElf(elf.get());
        scheduler->dispatch(*statement);

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

