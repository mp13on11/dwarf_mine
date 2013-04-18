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

#include <elves/quadratic_sieve/cuda/kernel.cuh>
#include <elves/cuda-utils/ErrorHandling.h>

extern __global__ void testAddKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testSubKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testMulKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testDivKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testModKernel(PNumData pLeft, PNumData pRight, PNumData output);
extern __global__ void testSmallerThanKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testLargerThanKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testLargerEqualKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testEqualKernel(PNumData pLeft, PNumData pRight, bool* output);
extern __global__ void testShiftLeftKernel(PNumData pLeft, uint32_t offset, PNumData output);
extern __global__ void testShiftRightKernel(PNumData pLeft, uint32_t offset, PNumData output);
extern __global__ void testModPowKernel(PNumData base, PNumData exponent, PNumData mod, PNumData result);
extern __global__ void testCudaPowKernel(int b, int e, int* result);

void testAdd(PNumData left, PNumData right, PNumData result)
{
    testAddKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testSub(PNumData left, PNumData right, PNumData result)
{
    testSubKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testMul(PNumData left, PNumData right, PNumData result)
{
    testMulKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testDiv(PNumData left, PNumData right, PNumData result)
{
    testDivKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testMod(PNumData left, PNumData right, PNumData result)
{
    testModKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testSmallerThan(PNumData left, PNumData right, bool* result)
{
    testSmallerThanKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testLargerThan(PNumData left, PNumData right, bool* result)
{
    testLargerThanKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testLargerEqual(PNumData left, PNumData right, bool* result)
{
    testLargerEqualKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testEqual(PNumData left, PNumData right, bool* result)
{
    testEqualKernel<<<1, 1>>>(left, right, result);
    CudaUtils::checkState();
}

void testShiftLeft(PNumData left,uint32_t offset, PNumData result)
{
    testShiftLeftKernel<<<1, 1>>>(left, offset, result);
    CudaUtils::checkState();
}

void testShiftRight(PNumData left,uint32_t offset, PNumData result)
{
    testShiftRightKernel<<<1, 1>>>(left, offset, result);
    CudaUtils::checkState();
}

void testModPow(PNumData base, PNumData exponent, PNumData mod, PNumData result)
{
    testModPowKernel<<<1, 1>>>(base, exponent, mod, result);
    CudaUtils::checkState();
}

void testCudaPow(int b, int e, int* result)
{
    testCudaPowKernel<<<1, 1>>>(b, e, result);
    CudaUtils::checkState();
}
