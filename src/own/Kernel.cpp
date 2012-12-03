#include "gold/Kernel.h"
#include "tools/MatrixHelper.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <math.h>
#include <omp.h>

using namespace std;

void printMatrix(const Matrix<float>& matrix) 
{
    std::cout << "[";
    for(size_t y=0; y<matrix.rows(); y++)
    {        
        if(y>0)
            std::cout << std::endl;
        std::cout << "[";
        for(size_t x=0; x<matrix.columns(); x++)
        {
            if(x>0)
                std::cout << " ";
            std::cout << matrix(y, x);
        }
        std::cout << "]";
    }
    std::cout << "]" << std::endl;
}

void Kernel::startup(const std::vector<std::string>& arguments)
{
    //for(const auto& arg : arguments)
    //std::cout << arg << std::endl;

    a = MatrixHelper::readMatrixFrom(arguments[0]);
    b = MatrixHelper::readMatrixFrom(arguments[1]);
    c = Matrix<float>(a.rows(), b.columns());
}
    
size_t div_ceil(size_t x, size_t y)
{
    return (x % y) ? (x / y + 1) : (x / y);
}

void multiplyBlock(Matrix<float>& left, Matrix<float>& right, Matrix<float>& outp, size_t workerCount, size_t index) {
        Matrix<float>& out = outp;
        //cout << "Launched by thread " << index << endl;

        size_t totalElements = out.rows() * out.columns();
        size_t elementsPerBlock = div_ceil(totalElements, workerCount);
        cout << "elementsPerBlock: " << elementsPerBlock << endl;

        size_t columnsPerBlock = (size_t)ceil(sqrt(elementsPerBlock));
        columnsPerBlock = min(columnsPerBlock, out.columns());

        size_t rowsPerBlock = div_ceil(elementsPerBlock, columnsPerBlock);
        cout << "rowsPerBlock: " << rowsPerBlock << endl;
        cout << "columnsPerBlock: " << columnsPerBlock << endl;

        size_t blocksPerRow = div_ceil(out.columns(), columnsPerBlock);
        size_t blocksPerColumns = div_ceil(out.rows(), rowsPerBlock);

        size_t rowStart = (index / blocksPerRow) * rowsPerBlock;
        size_t columnStart = (index % blocksPerRow) * columnsPerBlock;

        cout << "beginngin from (" << columnStart << "," << rowStart << ")" << endl;

        size_t columnEnd = min(columnStart+columnsPerBlock, out.columns());
        size_t rowEnd = min(rowStart+rowsPerBlock, out.rows());
        for(size_t y=columnStart; y<columnEnd; y++)
        {        
            for(size_t x=rowStart; x<rowEnd; x++)
            {
                float val = 0;
                for(size_t i=0; i<left.columns(); i++)
                {
                    val += left(y,i) * right(i,x);
                }
                //cout << "calculating out(" << y << "," << x << ") = " << val << endl;
                out(y,x) = val;
            }
        }
    }

void Kernel::run()
{
    size_t workerCount = 5;
    thread* workers = new thread[workerCount];

    //function<void(size_t)> workerMethod = bind(multiplyBlock, a, b, c, workerCount);

    //omp_set_num_threads(8);

    size_t elementsPerBlock = 10*10;

    size_t columnsPerBlock = (size_t)ceil(sqrt(elementsPerBlock));
    columnsPerBlock = min(columnsPerBlock, c.columns());

    size_t rowsPerBlock = div_ceil(elementsPerBlock, columnsPerBlock);

    size_t blocksPerRow = div_ceil(c.columns(), columnsPerBlock);
    size_t blocksPerColumns = div_ceil(c.rows(), rowsPerBlock);





    size_t blockCount = blocksPerRow * blocksPerColumns;
    //cout << "blockCount: " << blockCount << endl;

    #pragma omp parallel for
    for(size_t index=0; index<blockCount; index++)
    {
        size_t rowStart = (index / blocksPerRow) * rowsPerBlock;
        size_t columnStart = (index % blocksPerRow) * columnsPerBlock;

        size_t columnEnd = min(columnStart+columnsPerBlock, c.columns());
        size_t rowEnd = min(rowStart+rowsPerBlock, c.rows());

        //cout << rowStart << ":" << columnStart << " - " << rowEnd << ":" << columnEnd << endl;

        for(size_t y=rowStart; y<rowEnd; y++)
        {        
            for(size_t x=columnStart; x<columnEnd; x++)
            {
                float val = 0;
                for(size_t i=0; i<a.columns(); i++)
                {
                    val += a(y,i) * b(i,x);
                }
                //cout << "calculating c(" << y << "," << x << ") = " << val << endl;
                c(y,x) = val;
            }
        }

    }
//        workers[i] = thread(multiplyBlock, a, b, c, workerCount, i);

    //for(size_t i=0; i<workerCount; i++)
    //    workers[i].join();

    delete[] workers;

    /*for(size_t y=0; y<c.rows(); y++)
    {        
        for(size_t x=0; x<c.columns(); x++)
        {
            float val = 0;
            c(y,x) = 0;
            for(size_t i=0; i<a.columns(); i++)
            {
                val += a(y,i) * b(i,x);
            }
            c(y,x) = val;
        }
    }*/
}

void Kernel::shutdown(const std::string& outputFilename)
{
    //std::cout << outputFilename << std::endl;
    //printMatrix(c);
    MatrixHelper::writeMatrixTo(outputFilename, c);
}

std::shared_ptr<BenchmarkKernel> createKernel()
{
    return std::shared_ptr<BenchmarkKernel>(new Kernel());
}
