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
