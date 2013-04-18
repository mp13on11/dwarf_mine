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

#pragma once

#include "MatrixSlice.h"
#include "common/BenchmarkResults.h"
#include <list>
#include <vector>

class MatrixSlicer
{
public:
    typedef std::vector<MatrixSlice> SliceList;
    typedef std::list<NodeRating> RatingList;

    SliceList sliceAndDice(const BenchmarkResult& results, size_t rows, size_t columns) const;
    SliceList stripLayout(const BenchmarkResult& results, size_t rows, size_t columns) const; // WIP
    size_t testDeterminePivot(size_t rowsOrCols, const BenchmarkResult& results) const;
private:
    void sliceRows(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void sliceColumns(size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns) const;
    void setup(const BenchmarkResult& results) const;
    size_t determinePivot(size_t rowsOrCols) const;
    size_t processRating(size_t y, size_t x, size_t rows, size_t cols, bool colWise) const;

    size_t layoutStrip(const BenchmarkResult& results, size_t rows, size_t columns, size_t finishedIndex, bool verticalStrips) const; // WIP
    double computeAverageAspectRatio(BenchmarkResult& results, size_t finishedIndex, size_t numItems); // WIP
    int sizeForNode(BenchmarkResult& result); // WIP
    int computeSize(BenchmarkResult& results, size_t finishedIndex, size_t numItems); // WIP


    mutable SliceList slices;
    mutable RatingList ratings;
};
