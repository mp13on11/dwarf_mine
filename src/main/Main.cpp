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

const int BENCHMARK_ITERATIONS = 100;

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
      
        BenchmarkRunner preBenchmarkRunner(BENCHMARK_ITERATIONS);
        preBenchmarkRunner.runBenchmark(benchmarkStatement, *factory);
        auto weightedResults = preBenchmarkRunner.getWeightedResults();
       
        printResultOnMaster("Weighted", weightedResults);

        auto statement = config.createProblemStatement("matrix");

        BenchmarkRunner clusterBenchmarkRunner(BENCHMARK_ITERATIONS, weightedResults);
        clusterBenchmarkRunner.runBenchmark(*statement, *factory);
        auto clusterResults = clusterBenchmarkRunner.getTimedResults();

        printResultOnMaster("Cluster", clusterResults, "µs");

        if (MPI::COMM_WORLD.Get_rank() == MASTER)
        {
            BenchmarkRunner singleBenchmarkRunner(BENCHMARK_ITERATIONS, MASTER);
            singleBenchmarkRunner.runBenchmark(*statement, *factory);

            auto singleResults = singleBenchmarkRunner.getTimedResults();

            printResultOnMaster("Master", singleResults, "µs");
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

