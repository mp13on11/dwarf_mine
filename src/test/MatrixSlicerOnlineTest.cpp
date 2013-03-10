#include "MatrixSlicerOnlineTest.h"
#include "MatrixSlicerUtil.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <matrix/MatrixSlice.h>
#include <algorithm>
#include <cstddef>

using namespace std;

TEST_F(MatrixSlicerOnlineTest, EqualSlicing)
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

TEST_F(MatrixSlicerOnlineTest, RemainderRowSlicing)
{
    const size_t rows = 100;
    const size_t columns = 100;
    const auto slices = slicer.layout(rows, columns, 3, 1);   

    for (const auto& slice : slices)
        EXPECT_EQ(slice.getColumns(), (size_t)100);

    EXPECT_EQ(slices[0].getRows(),   (size_t)34);
    EXPECT_EQ(slices[0].getStartX(), (size_t)0);
    EXPECT_EQ(slices[0].getStartY(), (size_t)0);

    EXPECT_EQ(slices[1].getRows(),   (size_t)34);
    EXPECT_EQ(slices[1].getStartX(), (size_t)0);
    EXPECT_EQ(slices[1].getStartY(), (size_t)34);

    EXPECT_EQ(slices[2].getRows(),   (size_t)32);
    EXPECT_EQ(slices[2].getStartX(), (size_t)0);
    EXPECT_EQ(slices[2].getStartY(), (size_t)68);
}

TEST_F(MatrixSlicerOnlineTest, RemainderColumnSlicing)
{
    const size_t rows = 100;
    const size_t columns = 100;
    const auto slices = slicer.layout(rows, columns, 1, 3);   

    for (const auto& slice : slices)
        EXPECT_EQ(slice.getRows(), (size_t)100);

    EXPECT_EQ(slices[0].getColumns(), (size_t)34);
    EXPECT_EQ(slices[0].getStartX(),  (size_t)0);
    EXPECT_EQ(slices[0].getStartY(),  (size_t)0);

    EXPECT_EQ(slices[1].getColumns(), (size_t)34);
    EXPECT_EQ(slices[1].getStartX(),  (size_t)34);
    EXPECT_EQ(slices[1].getStartY(),  (size_t)0);

    EXPECT_EQ(slices[2].getColumns(), (size_t)32);
    EXPECT_EQ(slices[2].getStartX(),  (size_t)68);
    EXPECT_EQ(slices[2].getStartY(),  (size_t)0);
}
