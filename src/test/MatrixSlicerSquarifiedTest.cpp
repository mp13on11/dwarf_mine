#include "MatrixSlicerSquarifiedTest.h"
#include "MatrixSlicerUtil.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <matrix/MatrixSlice.h>
#include <algorithm>
#include <cstddef>

using namespace std;

typedef Matrix<float> TestGrid;

TEST_F(MatrixSlicerSquarifiedTest, SimpleUnifiedSlicingTest)
{
    size_t rows = 100;
    size_t columns = 100;
    size_t area = rows * columns;
    auto slices = slicer.layout({{0, 1.0}, {1, 1.0}, {2, 1.0}, {3, 1.0}}, rows, columns);
    
    size_t sliceArea = area / 4;
    verifySlices(slices, vector<size_t>{sliceArea, sliceArea, sliceArea, sliceArea});
    
}

TEST_F(MatrixSlicerSquarifiedTest, SimpleDifferentWeightSlicingTest)
{
    size_t rows = 100;
    size_t columns = 100;
    size_t area = rows * columns;
    auto slices = slicer.layout({{0, 6}, {1, 2}, {2, 1}, {3, 1}}, rows, columns);
    
    verifySlices(slices, vector<size_t>{ 6 * area / 10, 2 * area / 10, area / 10, area / 10});
}

TEST_F(MatrixSlicerSquarifiedTest, UnifiedSlicingTest)
{
    size_t rows = 33;
    size_t columns = 67;
    
    auto slices = slicer.layout({{0, 1.0}, {1, 1.0}, {2, 1.0}, {3, 1.0}}, rows, columns);
    
    verifySlices(slices, vector<size_t>{ 561 , 561, 561, 528});
}

TEST_F(MatrixSlicerSquarifiedTest, DifferentWeightSlicingTest)
{
    size_t rows = 33;
    size_t columns = 67;
    
    auto slices = slicer.layout({{0, 6}, {1, 2}, {2, 1.0}, {3, 1.0}}, rows, columns);

    verifySlices(slices, vector<size_t>{ 1353 , 200, 208, 450});
}

TEST_F(MatrixSlicerSquarifiedTest, SliceMultipleTest)
{
    size_t rows = 5;
    size_t columns = 5;
    
    auto slices = slicer.layout({{0, 1.0}, {1, 1.0}, {2, 1.0},{3, 1.0},{4, 1.0},{5, 1.0}, {6, 1.0}, {7, 1.0},}, rows, columns);
    bool matrix[5][5];
    for(size_t row = 0; row < rows; ++row)
    {
        for(size_t column = 0; column < columns; ++column)
        {
            matrix[row][column] = false;
        }
    }
    for(auto slice : slices)
    {
        ASSERT_NE((size_t) 0, rows);
        ASSERT_NE((size_t) 0, columns);
        for(size_t row = 0; row < slice.getRows(); ++row)
        {
            for(size_t column = 0; column < slice.getColumns(); ++column)
            {
                ASSERT_FALSE(matrix[slice.getStartY() + row][slice.getStartX() + column]);
                matrix[slice.getStartY() + row][slice.getStartX() + column] = true;
            }
        }
    }

    for(size_t row = 0; row < rows; ++row)
    {
        for(size_t column = 0; column < columns; ++column)
        {
            ASSERT_TRUE(matrix[row][column]);
        }
    }
}
