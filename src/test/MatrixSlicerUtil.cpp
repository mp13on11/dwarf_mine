#include "MatrixSlicerUtil.h"
#include <gtest/gtest.h>

using namespace std;

void verifySlice(MatrixSlice& slice, size_t x, size_t y, size_t columns, size_t rows)
{
    EXPECT_EQ((size_t)x, slice.getStartX());
    EXPECT_EQ((size_t)y, slice.getStartY());
    EXPECT_EQ(rows, slice.getRows());
    EXPECT_EQ(columns, slice.getColumns());
}

ostream& operator<<(ostream& stream, const MatrixSlice& slice)
{
    return stream
        << "("
        << slice.getStartX() << ", "
        << slice.getStartY() << ", "
        << slice.getColumns() << ", "
        << slice.getRows() << ")";
}

ostream& operator<<(ostream& stream, const SliceList& slices)
{
    stream << "[";
    bool first = true;
    for (const auto& slice : slices)
    {
        if (!first)
            stream << ", ";
        stream << slice;
        first = false;
    }
    stream << "]";

    return stream;
}