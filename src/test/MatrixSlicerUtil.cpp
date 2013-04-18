#include "MatrixSlicerUtil.h"
#include <gtest/gtest.h>

using namespace std;

void verifySlice(const MatrixSlice& slice, size_t x, size_t y, size_t columns, size_t rows)
{
    EXPECT_EQ((size_t)x, slice.getStartX());
    EXPECT_EQ((size_t)y, slice.getStartY());
    EXPECT_EQ(rows, slice.getRows());
    EXPECT_EQ(columns, slice.getColumns());
}
#include <iostream>
void verifySlices(const SliceList& slices, const AreaList& areas)
{
    EXPECT_EQ(areas.size(), slices.size());
    vector<bool> found(areas.size());
    //size_t j = 0;
    for (const auto& slice : slices)
    {
        // cout << slice << endl;
        // cout << areas[j++] << endl;
        for (size_t i = 0; i < areas.size(); ++i)
        {
            if (slice.getRows() * slice.getColumns() == areas[i] && !found[i])
            {
                found[i] = true;
                continue;
            }
        }
    }
    for (const auto& value : found)
    {
        EXPECT_EQ(true, value);
    }
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