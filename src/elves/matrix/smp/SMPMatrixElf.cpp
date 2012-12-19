#include "SMPMatrixElf.h"
#include <iostream>
#include <math.h>
#include <random>
#include <omp.h>
#include <functional>
#include "../Matrix.h"
#include "../MatrixHelper.h"

size_t div_ceil(size_t x, size_t y)
{
    return (x % y) ? (x / y + 1) : (x / y);
}

void SMPMatrixElf::run(std::istream& input, std::ostream& output)
{
    using namespace std;

    Matrix<float> a = MatrixHelper::readMatrixFrom(input);
    Matrix<float> b = MatrixHelper::readMatrixFrom(input);
  
    MatrixHelper::validateMultiplicationPossible(a, b);

    size_t leftRows = a.rows();
    size_t rightCols = b.columns();

    Matrix<float> c(leftRows, rightCols);

    size_t elementsPerBlock = 10*10;
    size_t columnsPerBlock = (size_t)ceil(sqrt(elementsPerBlock));
    columnsPerBlock = min(columnsPerBlock, c.columns());

    size_t rowsPerBlock = div_ceil(elementsPerBlock, columnsPerBlock);

    size_t blocksPerRow = div_ceil(c.columns(), columnsPerBlock);
    size_t blocksPerColumns = div_ceil(c.rows(), rowsPerBlock);

    size_t blockCount = blocksPerRow * blocksPerColumns;

    #pragma omp parallel for
    for(size_t index=0; index<blockCount; index++)
    {
        size_t rowStart = (index / blocksPerRow) * rowsPerBlock;
        size_t columnStart = (index % blocksPerRow) * columnsPerBlock;

        size_t columnEnd = min(columnStart+columnsPerBlock, c.columns());
        size_t rowEnd = min(rowStart+rowsPerBlock, c.rows());

        for(size_t y=rowStart; y<rowEnd; y++)
        {        
            for(size_t x=columnStart; x<columnEnd; x++)
            {
                float val = 0;
                for(size_t i=0; i<a.columns(); i++)
                {
                    val += a(y,i) * b(i,x);
                }
                c(y,x) = val;
            }
        }

    }
    MatrixHelper::writeMatrixTo(output, c);
}
