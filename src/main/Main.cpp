#include <iostream>
#include <cstdlib>
#include <mpi.h>

#include <matrix/cuda/CudaMatrixElf.h>
#include <matrix/smp/SMPMatrixElf.h>

using namespace std;

int main(int argc, char** argv) 
{
    MPI::Init(argc, argv);

    //cout << "Hello World from "<<getenv("OMPI_COMM_WORLD_LOCAL_RANK")<<" of "<<getenv("OMPI_COMM_WORLD_LOCAL_SIZE")<<" on "<<getenv("OMPI_COMM_WORLD_RANK")<<endl;

    CudaMatrixElf elf;
    elf.run(cin, cout);
 	SMPMatrixElf elf2;
    elf2.run(cin, cout);
    return 0;
}

