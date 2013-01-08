#include <iostream>
#include <string>
#include <fstream>

#include "BenchmarkRunner.h"
#include "Configuration.h"
#include "matrix/MatrixHelper.h"
#include "matrix/Matrix.h"
#include "MpiUtils.h"

using namespace std;

ostream& printResultOnMaster(ostream& o, string preamble, BenchmarkResult results, string unit = "")
{
    if (MPIGuard::isMaster())
    {
        o << preamble <<" "<<unit<< "\n" << results;
    }
    return o;
}

BenchmarkResult determineWeightedConfiguration(Configuration& config)
{
    auto factory = config.getElfFactory();
    auto statement = config.getProblemStatement();
    BenchmarkRunner runner(config);
    runner.runBenchmark(*statement, *factory);
    return runner.getWeightedResults(); 
}

void exportClusterConfiguration(const string& filename, BenchmarkResult& result)
{
    fstream file(filename, fstream::out);
    if (!file.is_open())
    {
        cerr << "ERROR: Could not write "<<filename<<endl;
        exit(1);
    }
    file << result;
    file.close();
}

BenchmarkResult importClusterConfiguration(const string& filename)
{
    fstream file(filename, fstream::in);
    if (!file.is_open())
    {
        cerr << "ERROR: Coult not read "<<filename<<endl;
        exit(1);
    }
    BenchmarkResult result;
    file >> result;
    file.close();
    if (result.size() != MPIGuard::numberOfNodes())
    {
        cerr << "ERROR: Number of nodes does not match configured number of nodes" <<endl;
        exit(1);
    }
    return result;
}

BenchmarkResult runTimedMeasurement(Configuration& config, BenchmarkResult& weightedResults)
{    
    auto factory = config.getElfFactory();
    auto statement = config.getProblemStatement(true);
    BenchmarkRunner runner(config, weightedResults);
    runner.runBenchmark(*statement, *factory);
    return runner.getTimedResults();
}

int main(int argc, char** argv)
{
    Configuration config(argc, argv);
    config.parseArguments(); 
    
    // used to ensure MPI::Finalize is called on exit of the application
    auto mpiGuard = MPIGuard(argc, argv);
    
    if (MPIGuard::isMaster())
    {
		cout << config <<endl;
	}
    
    try
    {
        BenchmarkResult weightedResults;
        if (config.exportConfiguration() || !config.importConfiguration())
        {
            cout << "Calculating node weights" <<endl;
            weightedResults = determineWeightedConfiguration(config);
            printResultOnMaster(cout, "Weighted", weightedResults);
        }
        if (config.exportConfiguration())
        {
            cout << "Exporting node weights" <<endl;
    		exportClusterConfiguration(config.getExportConfigurationFilename(), weightedResults);
		}
        if (config.importConfiguration())
        {
            cout << "Importing node weights" <<endl;
            weightedResults = importClusterConfiguration(config.getImportConfigurationFilename());
        }
        if (!config.skipBenchmark())
        {
            cout << "Running benchmark" <<endl;
            auto clusterResults = runTimedMeasurement(config, weightedResults);
            printResultOnMaster(cout, "Measured Time:", clusterResults, "µs");    
        }
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

