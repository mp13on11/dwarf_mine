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
    auto rows = 100;
    auto columns = 100;
    auto slices = slicer.layout({{0, 1}, {1, 1}, {2, 1}, {3, 1}}, rows, columns);
    ASSERT_EQ((size_t)4, slices.size());
    
    verifySlice(slices[0],  0,  0, 50, 50);
    verifySlice(slices[1], 50,  0, 50, 50);
    verifySlice(slices[2],  0, 50, 50, 50);
    verifySlice(slices[3], 50, 50, 50, 50);
}