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

#include "MatrixSlice.h"
#include <Matrix.h>
#include <cassert>

MatrixSlice::MatrixSlice(
    size_t startX,
    size_t startY,
    size_t columns,
    size_t rows
) :
    x(startX),
    y(startY),
    columns(columns),
    rows(rows)
{
}

MatrixSlice::MatrixSlice(
    int responsibleNode,
    size_t startX,
    size_t startY,
    size_t columns,
    size_t rows
) :
    nodeId(responsibleNode),
    x(startX),
    y(startY),
    columns(columns),
    rows(rows)
{
}
#include <iostream>
Matrix<float> MatrixSlice::extractSlice(const Matrix<float>& sourceMatrix, bool rowWise) const
{
    // short cut to prevent unnecessary copies
    if (x == 0 && y == 0 && columns == sourceMatrix.columns() && rows == sourceMatrix.rows())
    {
        return sourceMatrix;
    }
    size_t numRows, numColumns, yOffset, xOffset;
    if (rowWise)
    {
        xOffset = 0;
        yOffset = y;
        numRows = rows;
        numColumns = sourceMatrix.columns();
    }
    else
    {
        xOffset = x;
        yOffset = 0;
        numRows = sourceMatrix.rows();
        numColumns = columns;
    }

    Matrix<float> result(numRows, numColumns);

    for(size_t row = 0; row < numRows; ++row)
    {
        for(size_t column = 0; column < numColumns; ++column)
        {
            result(row, column) = sourceMatrix(row + yOffset, column + xOffset);
        }
    }

    return result;
}

void MatrixSlice::injectSlice(const Matrix<float>& sliceData, Matrix<float>& destMatrix) const
{
    using namespace std;
    assert(rows == sliceData.rows());
    assert(columns == sliceData.columns());

    for(size_t row = 0; row < rows; ++row)
    {
        for(size_t column = 0; column < columns; ++column)
        {
            destMatrix(row + y, column + x) = sliceData(row, column);
        }
    }
}

int MatrixSlice::getNodeId() const
{
    return nodeId;
}

void MatrixSlice::setNodeId(const int nodeId)
{
    this->nodeId = nodeId;
}

size_t MatrixSlice::getStartX() const
{
    return x;
}

size_t MatrixSlice::getStartY() const
{
    return y;
}

size_t MatrixSlice::getColumns() const
{
    return columns;
}

size_t MatrixSlice::getRows() const
{
    return rows;
