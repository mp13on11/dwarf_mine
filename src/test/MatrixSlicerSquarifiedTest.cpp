#include "MatrixSlicerSquarifiedTest.h"
#include "MatrixSlicerUtil.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <matrix/MatrixSlice.h>
#include <algorithm>
#include <cstddef>

const size_t ROWS = 910;
const size_t COLS = 735;
const size_t NODE_COUNTS[] = { 1, 2, 3, 5, 7, 12, 25, 80, 110, 127 };

using namespace std;

typedef Matrix<float> TestGrid;

TEST_F(MatrixSlicerSquarifiedTest, SimpleUnifiedSlicingTest)
{
    size_t rows = 100;
    size_t columns = 100;
    size_t area = rows * columns;
    auto slices = slicer.layout({{0, 1}, {1, 1}, {2, 1}, {3, 1}}, rows, columns);
    
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
    
    auto slices = slicer.layout({{0, 1}, {1, 1}, {2, 1}, {3, 1}}, rows, columns);
    
    verifySlices(slices, vector<size_t>{ 561 , 561, 561, 528});
}

TEST_F(MatrixSlicerSquarifiedTest, DifferentWeightSlicingTest)
{
    size_t rows = 33;
    size_t columns = 67;
    
    auto slices = slicer.layout({{0, 6}, {1, 2}, {2, 1}, {3, 1}}, rows, columns);
    
    verifySlices(slices, vector<size_t>{ 1353 , 200, 208, 425});
}
