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

#include <cstddef>

template<typename T>
class Matrix;

class MatrixSlice
{
public:
    MatrixSlice() = default;
    MatrixSlice(
        std::size_t startX,
        std::size_t startY,
        std::size_t columns,
        std::size_t rows
    );
    MatrixSlice(
        int responsibleNode,
        std::size_t startX,
        std::size_t startY,
        std::size_t columns,
        std::size_t rows
    );

    Matrix<float> extractSlice(const Matrix<float>& sourceMatrix, bool rowWise) const;
    void injectSlice(const Matrix<float>& sliceData, Matrix<float>& destMatrix) const;

    int getNodeId() const;
    void setNodeId(const int nodeId);
    std::size_t getStartX() const;
    std::size_t getStartY() const;
    std::size_t getColumns() const;
    std::size_t getRows() const;

private:
    int nodeId;
    std::size_t x;
    std::size_t y;
    std::size_t columns;
    std::size_t rows;
