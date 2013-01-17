#include "MatrixSlicingTest.h"
#include "MatrixSlicerUtil.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>
#include <matrix/MatrixSlice.h>
#include <algorithm>
#include <cstddef>

const size_t ROWS = 910;
const size_t COLS = 735;
const size_t NODE_COUNTS[] = { 1, 2, 3, 5, 7, 12, 25, 80, 110, 127 };

typedef Matrix<float> TestGrid;

using namespace std;
typedef MatrixSlicer::SliceList SliceList;

BenchmarkResult makeUniformRatings(size_t number)
{
    BenchmarkResult ratings;
    for (size_t i=0; i<number; ++i)
    {
        ratings[i] = number;
    }
    return ratings;
}

TestGrid setupTestGrid(size_t rows, size_t cols)
{
    TestGrid testGrid(rows, cols);
    MatrixHelper::fill(testGrid, [](){ return 0.0f; });
    return testGrid;
}

string makeErrorMsg(size_t nodeCount, const MatrixSlice& slice)
{
    stringstream errorMsg;
    errorMsg
        << " out of bounds. Nodes: "
        << nodeCount << ", Slice: " << slice;
    return errorMsg.str();
}

void applySliceToTestGrid(const MatrixSlice& slice, TestGrid* grid, size_t nodeCount)
{
    for (size_t i=0; i<slice.getRows(); ++i)
    {
        for (size_t j=0; j<slice.getColumns(); ++j)
        {
            auto row = i + slice.getStartY();
            auto col = j + slice.getStartX();
            ASSERT_LT(row, grid->rows()) << "Row" << makeErrorMsg(nodeCount, slice);
            ASSERT_LT(col, grid->columns()) << "Column" << makeErrorMsg(nodeCount, slice);
            (*grid)(row, col) += 1.0f;
        }
    }
}

TestGrid fillTestGrid(size_t rows, size_t cols, const SliceList& slices, const BenchmarkResult& ratings)
{
    TestGrid testGrid = setupTestGrid(rows, cols);
    for (const MatrixSlice& slice : slices)
    {
        applySliceToTestGrid(slice, &testGrid, ratings.size());
        if (testing::Test::HasFatalFailure()) break;
    }

    return testGrid;
}

TestGrid makeTestGrid(size_t nodeCount, const MatrixSlicer& slicer)
{
    auto ratings = makeUniformRatings(nodeCount);
    auto slices = slicer.sliceAndDice(ratings, ROWS, COLS);
    return fillTestGrid(ROWS, COLS, slices, ratings);
}

void checkTestGrid(const TestGrid& testGrid, const SliceList& slices)
{
    for(size_t i=0; i<ROWS; ++i)
    {
        for (size_t j=0; j<COLS; ++j)
        {
            ASSERT_GT(testGrid(i, j), 0.0f)
                << "Slices incomplete at (" << i
                << ", " << j << "). Slices: " << slices;

            ASSERT_LT(testGrid(i, j), 2.0f)
                << "Slices overlap at (" << i
                << ", " << j << "). Slices: " << slices;
        }
    }
}

TEST_F(MatrixSlicingTest, PivotMatrixSliceTest)
{
	auto slices = slicer.sliceAndDice({{0, 10}, {1, 10}, {2, 80}}, 100, 100);
	ASSERT_EQ(slices.size(), (size_t)3);
	
	verifySlice(slices[0], 0, 0, 45, 100);
	verifySlice(slices[1], 45, 0, 55, 82);
	verifySlice(slices[2], 45, 82, 55, 18);
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

TEST_F(MatrixSlicingTest, ValidSlicesTest)
{
    for (size_t count : NODE_COUNTS)
    {
        auto ratings = makeUniformRatings(count);
        auto slices = slicer.sliceAndDice(ratings, ROWS, COLS);
        TestGrid testGrid = fillTestGrid(ROWS, COLS, slices, ratings);
        if (HasFatalFailure()) return;

        checkTestGrid(testGrid, slices);
        if (HasFatalFailure()) return;
    }
}
