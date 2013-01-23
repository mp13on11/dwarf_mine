#include "SimpleConfiguration.h"
#include "common/BenchmarkRunner.h"
#include "common/MpiGuard.h"

#include <exception>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    try
    {
        MpiGuard guard(argc, argv);
        SimpleConfiguration config(argc, argv);

        if (config.shouldPrintHelp())
        {
            SimpleConfiguration::printHelp();
            return 0;
        }

        cout << config << endl;

        BenchmarkResult rating;
        rating.insert({0, 1});
        BenchmarkRunner runner(config, rating);
        auto statement = config.createProblemStatement(false);
        auto factory = config.createSchedulerFactory();
        runner.runBenchmark(*statement, *factory);
    }
    catch (const boost::program_options::error& e)
    {
        SimpleConfiguration::printHelp();
        cerr << e.what() << endl;
        return 1;
    }
    catch (const exception& e)
    {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "FATAL ERROR of unknown type" << endl;
        return 1;
    }
    return 0;
}
