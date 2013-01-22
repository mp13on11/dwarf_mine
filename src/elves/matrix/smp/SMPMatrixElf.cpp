#include "SMPMatrixElf.h"
#include <iostream>
#include <math.h>
#include <random>
#include <omp.h>
#include <functional>
#include "common/Utils.h"
#include "Matrix.h"

SMPMatrixElf::MatrixT SMPMatrixElf::multiply(const MatrixT& left, const MatrixT& right)
{
    using namespace std;

    size_t leftRows = left.rows();
    size_t rightCols = right.columns();

    MatrixT c(leftRows, rightCols);

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
                for(size_t i=0; i<left.columns(); i++)
                {
                    val += left(y,i) * right(i,x);
                }
                c(y,x) = val;
            }
        }

    }

    return c;
}
