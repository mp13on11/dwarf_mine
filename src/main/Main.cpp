#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <sstream>

#include "matrix/cuda/CudaMatrixElf.h"
#include "matrix/smp/SMPMatrixElf.h"
#include "matrix/MatrixHelper.h"
#include "CudaElfFactory.h"
#include "SMPElfFactory.h"
#include "BenchmarkRunner.h"

using namespace std;

int main(int argc, char** argv) 
{
    MPI::Init(argc, argv);

    Matrix<float> first(10,10);
    Matrix<float> second(10, 10);
    stringstream in;
    stringstream out;
    MatrixHelper::writeMatrixTo(in, first);
    MatrixHelper::writeMatrixTo(in, second);
    ProblemStatement statement {in, out, "matrix"};
    
    BenchmarkRunner runner(1);
    runner.runBenchmark(statement, CudaElfFactory());
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

