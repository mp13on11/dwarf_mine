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
#include "matrix/smp/SMPMatrixElf.h"
#include "matrix/MatrixHelper.h"
#include "matrix/Matrix.h"

using namespace std;

void generateProblemData(stringstream& in, stringstream& out)
{
    Matrix<float> first(100,100);
    Matrix<float> second(100, 100);
    auto distribution = uniform_real_distribution<float> (-100, +100);
    auto engine = mt19937(time(nullptr));
    auto generator = bind(distribution, engine);
    MatrixHelper::fill(first, generator);
    MatrixHelper::fill(second, generator);
    MatrixHelper::writeMatrixTo(in, first);
    MatrixHelper::writeMatrixTo(in, second);
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
        stringstream in;
        stringstream out;
        generateProblemData(in, out);
        ProblemStatement statement{ in, out, "matrix"};

        unique_ptr<ElfFactory> factory(config.getElfFactory(statement.elfCategory));
       

        BenchmarkRunner runner(100);
        runner.runBenchmark(statement, *factory);
        auto results = runner.getResults();
        for (const auto& result: results)
        {
            cout << result.first << " - " <<result.second<<endl;
        }
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

