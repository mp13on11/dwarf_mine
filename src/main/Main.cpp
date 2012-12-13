#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <sstream>

#include "matrix/cuda/CudaMatrixElf.h"
#include "matrix/smp/SMPMatrixElf.h"
#include "CudaElfFactory.h"
#include "SMPElfFactory.h"
#include "BenchmarkRunner.h"

using namespace std;

int main(int argc, char** argv) 
{
    MPI::Init(argc, argv);

    stringstream in;
    stringstream out;
    ProblemStatement statement {in, out, "matrix"};
    
    BenchmarkRunner runner(1);
    runner.runBenchmark(statement, SMPElfFactory());
    auto results = runner.getResults();
    for (auto& result: results)
    {
        cout << result.first << " - " <<result.second<<endl;
    }
    
  //   CudaMatrixElf elf;
  //   elf.run(cin, cout);
 	 // SMPMatrixElf elf2;
   //   elf2.run(cin, cout);
    MPI::Finalize();
    return 0;
}

