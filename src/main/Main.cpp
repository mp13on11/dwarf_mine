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
#include "MpiUtils.h"

using namespace std;

const int BENCHMARK_ITERATIONS = 10;
const int PRE_BENCHMARK_MATRIX_SIZE = 500;

void generateProblemData(ProblemStatement& statement)
{
    Matrix<float> first(PRE_BENCHMARK_MATRIX_SIZE,PRE_BENCHMARK_MATRIX_SIZE);
    Matrix<float> second(PRE_BENCHMARK_MATRIX_SIZE, PRE_BENCHMARK_MATRIX_SIZE);
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

void printResultOnMaster(string preamble, BenchmarkResult results, string unit = "")
{
    if (MPI::COMM_WORLD.Get_rank() == MASTER)
    {
        cout << preamble << "\n";
        for (const auto& result : results)
        {
            cout << result.first << " " << result.second << " "<< unit << "\n";
        }
        cout << flush;
    }
}

BenchmarkResult calculateWeightings(const ElfFactory& factory, const Configuration& config){
    ProblemStatement benchmarkStatement(config.getElfCategory());
    generateProblemData(benchmarkStatement);
      
    BenchmarkRunner preBenchmarkRunner(BENCHMARK_ITERATIONS);
    preBenchmarkRunner.runBenchmark(benchmarkStatement, factory);
    return preBenchmarkRunner.getWeightedResults();   
}

int main(int argc, char** argv)
{
    Configuration config(argc, argv);
    config.parseArguments(); 

    // used to ensure MPI::Finalize is called on exit of the application
    auto mpiGuard = MPIGuard(argc, argv);
    try
    {
        unique_ptr<ElfFactory> factory(config.getElfFactory());
        auto weightedResults = calculateWeightings(*factory, config);
        printResultOnMaster("Weighted", weightedResults);

        auto statement = config.createProblemStatement(config.getElfCategory());

        BenchmarkRunner clusterBenchmarkRunner(config, weightedResults);
        clusterBenchmarkRunner.runBenchmark(*statement, *factory);
        auto clusterResults = clusterBenchmarkRunner.getTimedResults();

        printResultOnMaster("Measured Time:", clusterResults, "µs");

    }
    catch (const exception &e)
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

