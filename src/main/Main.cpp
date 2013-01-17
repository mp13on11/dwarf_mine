#include "common/BenchmarkRunner.h"
#include "common/Configuration.h"
#include "common/MpiGuard.h"
#include "common/MpiHelper.h"
#include "matrix/MatrixHelper.h"
#include "matrix/Matrix.h"

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

BenchmarkResult determineWeightedConfiguration(const Configuration& config)
{
    auto factory = config.createSchedulerFactory();
    auto statement = config.createProblemStatement(true);
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

BenchmarkResult runTimedMeasurement(const Configuration& config, BenchmarkResult& weightedResults)
{
    auto factory = config.createSchedulerFactory();
    auto statement = config.createProblemStatement();
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


    try
    {
        Configuration config(argc, argv, MpiHelper::isMaster());

        if (!config.shouldBeVerbose() && (config.shouldBeQuiet() || !MpiHelper::isMaster()))
            silenceOutputStreams(true);

        cout << config <<endl;

        BenchmarkResult weightedResults;
        if (config.shouldExportConfiguration() || !config.shouldImportConfiguration())
        {
            cout << "Calculating node weights" <<endl;
            weightedResults = determineWeightedConfiguration(config);
            cout << "Weighted " << endl << weightedResults;
        }
        if (config.shouldExportConfiguration())
        {
            cout << "Exporting node weights" <<endl;
            exportClusterConfiguration(config.exportConfigurationFilename(), weightedResults);
        }
        if (config.shouldImportConfiguration())
        {
            cout << "Importing node weights" <<endl;
            weightedResults = importClusterConfiguration(config.importConfigurationFilename());
        }
        if (!config.shouldSkipBenchmark())
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

