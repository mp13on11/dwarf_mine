#include "MatrixSlicerOnlineTest.h"
#include "MatrixSlicerUtil.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <matrix/MatrixSlice.h>
#include <algorithm>
#include <cstddef>

using namespace std;

TEST_F(MatrixSlicerOnlineTest, QuadraticEqualSlicing)
{
    const size_t rows = 100;
    const size_t columns = 100;
    const auto slices = slicer.layout(rows, columns, 2, 2);   
    for (const auto& slice : slices)
    {
        EXPECT_EQ(slice.getRows(), (size_t)50);
        EXPECT_EQ(slice.getColumns(), (size_t)50);
    }
    EXPECT_EQ(slices[0].getStartX(), (size_t)0);
    EXPECT_EQ(slices[0].getStartY(), (size_t)0);
    EXPECT_EQ(slices[1].getStartX(), (size_t)0);
    EXPECT_EQ(slices[1].getStartY(), (size_t)50);
    EXPECT_EQ(slices[2].getStartX(), (size_t)50);
    EXPECT_EQ(slices[2].getStartY(), (size_t)0);
    EXPECT_EQ(slices[3].getStartX(), (size_t)50);
    EXPECT_EQ(slices[3].getStartY(), (size_t)50);
}
