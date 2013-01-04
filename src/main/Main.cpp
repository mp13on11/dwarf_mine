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
const int PRE_BENCHMARK_MATRIX_SIZE = 1500;

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

BenchmarkResult calculateWeightings(const ElfFactory& factory, Configuration& config){
	if (config.preBenchmark())
	{
		ProblemStatement benchmarkStatement(config.getElfCategory());
		generateProblemData(benchmarkStatement);
		  
		BenchmarkRunner preBenchmarkRunner(config);
		preBenchmarkRunner.runBenchmark(benchmarkStatement, factory);
		printResultOnMaster("Timed", preBenchmarkRunner.getTimedResults(), "µs");
		return preBenchmarkRunner.getWeightedResults();   
	}
	else
	{
		BenchmarkResult result;
		for (int i = 0; i < MPI::COMM_WORLD.Get_size(); ++i)
		{
			result[i] = 1;
		}
		return result;
	}
}


int main(int argc, char** argv)
{
    Configuration config(argc, argv);
    config.parseArguments(); 
    
    // used to ensure MPI::Finalize is called on exit of the application
    auto mpiGuard = MPIGuard(argc, argv);
    
    if (MPI::COMM_WORLD.Get_rank() == MASTER)
    {
		cout << config <<endl;
	}
    
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

