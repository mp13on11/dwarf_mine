/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#include "Matrix.h"
#include "SMPMatrixElf.h"
#include "common/Utils.h"

#include <cmath>
#include <functional>
#include <random>

using namespace std;

SMPMatrixElf::MatrixT SMPMatrixElf::multiply(const MatrixT& left, const MatrixT& right)
{
    if (left.empty() || right.empty()) return MatrixT();
    MatrixT c(left.rows(), right.columns());

    size_t elementsPerBlock = 10*10;
    size_t columnsPerBlock = (size_t)ceil(sqrt(elementsPerBlock));
    columnsPerBlock = min(columnsPerBlock, c.columns());

    size_t rowsPerBlock = div_ceil(elementsPerBlock, columnsPerBlock);

    size_t blocksPerRow = div_ceil(c.columns(), columnsPerBlock);
    size_t blocksPerColumns = div_ceil(c.rows(), rowsPerBlock);

    size_t blockCount = blocksPerRow * blocksPerColumns;

    MatrixT transposed = right.transposed();

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
                    val += left(y,i) * transposed(x,i);
                }
                c(y,x) = val;
            }
        }

    }

    return c;
