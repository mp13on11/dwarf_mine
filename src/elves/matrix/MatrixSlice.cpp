#include "MatrixSlice.h"
#include <Matrix.h>
#include <cassert>

MatrixSlice::MatrixSlice(
    NodeId responsibleNode,
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

void MatrixSlice::send() const
{

}

void MatrixSlice::receive() const
{

}

Matrix<float> MatrixSlice::extractSlice(const Matrix<float>& sourceMatrix, bool rowWise) const
{
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

NodeId MatrixSlice::getNodeId() const
{
    return nodeId;
}

size_t MatrixSlice::getColumns() const
{
    return columns;
}

size_t MatrixSlice::getRows() const
{
    return rows;
}
