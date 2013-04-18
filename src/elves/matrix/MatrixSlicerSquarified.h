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
#include <vector>
#include <list>

class MatrixSlicerSquarified
{

    public:
        typedef std::vector<MatrixSlice> SliceList;
        typedef std::list<NodeRating> RatingList;

        SliceList layout(const BenchmarkResult& results, size_t rows, size_t columns);

    private:
        SliceList _slices;
        MatrixSlice _unlayoutedSlice;
        std::list<NodeRating> _unlayoutedRatings;

        double calculateRatio(size_t smallestSide, RatingList strip);
        double calculateRatio(size_t smallestSide, RatingList strip, NodeRating head);

        size_t getSmallestSide();
        size_t getLongestSide();
        bool isVerticalStrip();
        size_t getX();
        size_t getY();
        void setUnlayoutedSlice(const std::list<NodeRating>& stripRatings, size_t x, size_t y, size_t rows, size_t columns);

        void addToLayout(std::list<NodeRating> stripRatings);
        void squarify(std::list<NodeRating>& strip);

        void setup(const BenchmarkResult& results, size_t area);

