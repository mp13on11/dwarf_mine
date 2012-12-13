#include <iostream>

#include <matrix/cuda/CudaMatrixElf.h>
#include <matrix/smp/SMPMatrixElf.h>

using namespace std;

int main(int argc, char** argv) 
{
    //CudaMatrixElf elf;
    //elf.test();
    SMPMatrixElf elf2;
    elf2.test();
    return 0;
}

