#include "BenchmarkRunner.h"
#include "Configuration.h"
#include "MpiGuard.h"
#include "MpiHelper.h"
#include "matrix/MatrixHelper.h"
#include "matrix/Matrix.h"

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

BenchmarkResult determineWeightedConfiguration(Configuration& config)
{
    auto factory = config.getElfFactory();
    auto statement = config.getProblemStatement(true);
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
        cerr << "ERROR: Could not read "<<filename<<endl;
        exit(1);
    }
    BenchmarkResult result;
    file >> result;
    file.close();
    if (result.size() != MpiHelper::numberOfNodes())
    {
        cerr << "ERROR: Number of nodes does not match configured number of nodes" <<endl;
        exit(1);
    }
    return result;
}

BenchmarkResult runTimedMeasurement(Configuration& config, BenchmarkResult& weightedResults)
{
    auto factory = config.getElfFactory();
    auto statement = config.getProblemStatement();
    BenchmarkRunner runner(config, weightedResults);
    runner.runBenchmark(*statement, *factory);
    return runner.getTimedResults();
}

void silenceOutputStreams(bool keepErrorStreams = false)
{
    cout.rdbuf(nullptr);

    if (!keepErrorStreams)
    {
        cerr.rdbuf(nullptr);
        clog.rdbuf(nullptr);
    }
}

int main(int argc, char** argv)
{
    // used to ensure MPI::Finalize is called on exit of the application
    MpiGuard guard(argc, argv);

    // Disable output on slaves
    if (!MpiHelper::isMaster())
        silenceOutputStreams();

    try
    {
        Configuration config(argc, argv);

        if (!config.parseArguments())
            return 2;

        if (config.getQuiet())
            silenceOutputStreams(true);

        cout << config <<endl;

        BenchmarkResult weightedResults;
        if (config.exportConfiguration() || !config.importConfiguration())
        {
            cout << "Calculating node weights" <<endl;
            weightedResults = determineWeightedConfiguration(config);
            cout << "Weighted " << endl << weightedResults;
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
            cout << "Measured Time: µs" << endl << clusterResults;
        }
    }
    catch (const logic_error& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
    catch (const exception& e)
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

