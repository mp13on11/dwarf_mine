#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mpi.h>
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

int main(int argc, char** argv) 
{
    vector<string> arguments(argv + 1, argv + argc);

    if (arguments.size() < 1)
    {
        cerr << "Usage: " << argv[0] << " cuda|smp" << endl;
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

    Matrix<float> first(10,10);
    Matrix<float> second(10, 10);
    stringstream in;
    stringstream out;
    MatrixHelper::writeMatrixTo(in, first);
    MatrixHelper::writeMatrixTo(in, second);
    ProblemStatement statement {in, out, "matrix"};
    
    try
    {
        BenchmarkRunner runner(1);
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

