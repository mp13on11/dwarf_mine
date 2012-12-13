#include <cstdlib>
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

#include "BenchmarkRunner.h"
#include "CudaElfFactory.h"
#include "SMPElfFactory.h"
#include "matrix/smp/SMPMatrixElf.h"
#include "matrix/MatrixHelper.h"

using namespace std;

void generateProblemData(stringstream& in, stringstream& out)
{
    Matrix<float> first(100,100);
    Matrix<float> second(100, 100);
    auto distribution = uniform_real_distribution<float> (-100, +100);
    auto engine = mt19937(time(0));
    auto generator = bind(distribution, engine);
    MatrixHelper::fill(first, generator);
    MatrixHelper::fill(second, generator);
    MatrixHelper::writeMatrixTo(in, first);
    MatrixHelper::writeMatrixTo(in, second);
}

int main(int argc, char** argv) 
{
    vector<string> arguments(argv + 1, argv + argc);

    if (arguments.size() < 3)
    {
        cerr << "Usage: " << argv[0] << " cuda|smp <left_matrix> <right_matrix>" << endl;
        return 1;
    }

    unique_ptr<ElfFactory> factory;

    if (arguments[0] == "cuda")
    {
        factory.reset(new CudaElfFactory());
    }
    else if (arguments[0] == "smp")
    {
        factory.reset(new SMPElfFactory());
    }
    else
    {
        cerr << "Usage: " << argv[0] << " cuda|smp" << endl;
        return 1;
    }

    MPI::Init(argc, argv);


    stringstream in;
    stringstream out;

    generateProblemData(in, out);
    ProblemStatement statement{ in, out, "matrix"};// = generateProblemStatement();
    
    try
    {
        BenchmarkRunner runner(100);
        runner.runBenchmark(statement, *factory);
        auto results = runner.getResults();
        for (auto& result: results)
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
    
  //   CudaMatrixElf elf;
  //   elf.run(cin, cout);
 	 // SMPMatrixElf elf2;
   //   elf2.run(cin, cout);
    MPI::Finalize();
    return 0;
}

