/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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
