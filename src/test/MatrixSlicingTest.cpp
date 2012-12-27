#include "MatrixSlicingTest.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <matrix/MatrixSlice.h>
#include <cstddef>

const size_t ROWS = 910;
const size_t COLS = 735;

const size_t NODE_COUNTS[] = { 1, 2, 3, 5, 7, 12, 25, 80, 110 };

using namespace std;

BenchmarkResult makeUniformRatings(size_t number)
{
    BenchmarkResult ratings;
    for (size_t i=0; i<number; ++i)
    {
        ratings[i] = 1/number * 100;
    }
    return ratings;
}

TEST_F(MatrixSlicingTest, SingleNodeKeepsWholeMatrixTest)
{
    auto slices = slicer.sliceAndDice({{0, 0}}, ROWS, COLS);
    ASSERT_EQ(slices.size(), (size_t)1);

    auto slice = slices[0];
    EXPECT_EQ(slice.getStartX(), (size_t)0);
    EXPECT_EQ(slice.getStartY(), (size_t)0);
    EXPECT_EQ(slice.getRows(), ROWS);
    EXPECT_EQ(slice.getColumns(), COLS);
}

TEST_F(MatrixSlicingTest, OneSlicePerNodeTest)
{
    for (size_t count : NODE_COUNTS)
    {
        auto slices = slicer.sliceAndDice(makeUniformRatings(count), ROWS, COLS);
        EXPECT_EQ(slices.size(), count);
    }
}

TEST_F(MatrixSlicingTest, SlicingIsDisjointAndCompleteTest)
{
    Matrix<float> testGrid(ROWS, COLS);
    MatrixHelper::fill(testGrid, [](){ return 0.0f; });

    for (size_t count : NODE_COUNTS)
    {
        auto slices = slicer.sliceAndDice(makeUniformRatings(count), ROWS, COLS);
        for (const MatrixSlice& slice : slices)
        {
            for (size_t i=0; i<slice.getRows(); ++i)
            {
                for (size_t j=0; j<slice.getColumns(); ++j)
                {
                    auto row = i + slice.getStartY();
                    auto col = j + slice.getStartX();
                    ASSERT_LT(row, testGrid.rows())
                        << "Row out of bounds. Nodes: "
                        << count << ", Slice: ("
                        << slice.getStartX() << ", "
                        << slice.getStartY() << ", "
                        << slice.getRows() << ", "
                        << slice.getColumns() << ")";
                    ASSERT_LT(col, testGrid.columns())
                        << "Column out of bounds. Nodes: "
                        << count << ", Slice: ("
                        << slice.getStartX() << ", "
                        << slice.getStartY() << ", "
                        << slice.getRows() << ", "
                        << slice.getColumns() << ")";
                    testGrid(row, col) += 1.0f;
                }
            }

        }
    }

    for (size_t i=0; i<ROWS*COLS; ++i)
    {
        float value = testGrid.buffer()[i];
        ASSERT_EQ(value, 1.0f);
    }
}
