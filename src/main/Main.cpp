#include "common/BenchmarkRunner.h"
#include "common/CommandLineConfiguration.h"
#include "common/MpiGuard.h"
#include "common/MpiHelper.h"
#include "matrix/MatrixHelper.h"
#include "matrix/Matrix.h"

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

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
        CommandLineConfiguration config(argc, argv);

        if (config.shouldPrintHelp())
        {
            config.printHelp();
            return 0;
        }

        if (!config.shouldBeVerbose() && (config.shouldBeQuiet() || !MpiHelper::isMaster()))
            silenceOutputStreams(true);

        cout << config << endl;

        BenchmarkRunner runner(config);
        BenchmarkResult nodeWeights;
        if (config.shouldExportConfiguration() || !config.shouldImportConfiguration())
        {
            cout << "Calculating node weights" <<endl;
            nodeWeights = runner.benchmarkIndividualNodes();
            cout << "Weighted " << endl << nodeWeights;
        }
        if (config.shouldExportConfiguration())
        {
            cout << "Exporting node weights" <<endl;
            exportClusterConfiguration(config.exportConfigurationFilename(), nodeWeights);
        }
        if (config.shouldImportConfiguration())
        {
            cout << "Importing node weights" <<endl;
            nodeWeights = importClusterConfiguration(config.importConfigurationFilename());
        }
        if (!config.shouldSkipBenchmark())
        {
            cout << "Running benchmark" <<endl;
            auto clusterResults = runner.runBenchmark(nodeWeights);
            cout << "Measured Times: µs" << endl;

            for (const auto& measurement : clusterResults)
                cout << "\t" << measurement.count() << endl;
        }
    }
    catch (const boost::program_options::error& e)
    {
        CommandLineConfiguration::printHelp();
        cerr << e.what() << endl;
        return 1;
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

