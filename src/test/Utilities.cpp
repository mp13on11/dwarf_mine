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

#include "Utilities.h"
#include <matrix/Matrix.h>
#include <matrix/MatrixHelper.h>

#include <cmath>
#include <algorithm>

const float DIFFERENCE_THRESHOLD = 1e-2f;
const float RELATIVE_ERROR_THRESHOLD = 1e-10;

using namespace std;

testing::AssertionResult AreMatricesEquals(Matrix<float> expected, Matrix<float>actual, float delta)
{
    if(expected.rows() != actual.rows())
    {
        return testing::AssertionFailure() << "different number of rows: " << expected.rows() << " != " << actual.rows();
    }
    if(expected.columns() != actual.columns())
    {
        return testing::AssertionFailure() << "different number of columns: " << expected.columns() << " != " << actual.columns();
    }
    for(size_t y = 0; y<expected.rows(); y++)
    {
        for(size_t x = 0; x<expected.columns(); x++)
        {
            float expectedVal = expected(y,x);
            float actualVal = actual(y,x);
            float maxVal = max(fabs(expectedVal), fabs(actualVal));
            float error = fabs(expectedVal - actualVal);

            if (maxVal >= RELATIVE_ERROR_THRESHOLD)
                error /= maxVal;

            if(error > delta)
            {
                // dump matrices
                MatrixHelper::writeMatrixTo("dump_expected.txt", expected);
                MatrixHelper::writeMatrixTo("dump_actual.txt", actual);

                // return failure
                return testing::AssertionFailure()
                << "The relative error at (" << y << "," << x << ") is " << error << ", which exceeds " << delta << ", where" << endl
                << "expected(y,x) = " << expectedVal << " and" << endl
                << "actual(y,x) = " << actualVal << ".";
            }
        }
    }
    return testing::AssertionSuccess();
}

testing::AssertionResult AreMatricesEquals(Matrix<float> a, Matrix<float>b)
{
    return AreMatricesEquals(a, b, DIFFERENCE_THRESHOLD);
