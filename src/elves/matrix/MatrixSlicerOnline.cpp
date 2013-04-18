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

#include "MatrixSlicerOnline.h"
#include "MatrixSlice.h"
#include <cmath>
#include <iostream>

using namespace std;

vector<MatrixSlice> MatrixSlicerOnline::layout(size_t rows, size_t columns, size_t rowparts, size_t colparts)
{
    slices.clear();
    size_t slice_rows = ceil((double)rows / rowparts);
    size_t slice_cols = ceil((double)columns / colparts);
    for(size_t i=0; i<columns; i+= slice_cols){
        for(size_t j=0; j<rows; j+= slice_rows){
            slices.push_back(MatrixSlice{
                i,
                j,
                i + slice_cols <= columns ? slice_cols : slice_cols - (i + slice_cols - columns),
                j + slice_rows <= rows ? slice_rows : slice_rows - (j + slice_rows - rows)
            });
        }
    }
    return slices;
}

