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

#include "OthelloCudaUtilTest.h"
#include <cuda-utils/Memory.h>
#include <vector>
#include <tuple>

using namespace std;


TEST_F(OthelloCudaUtilTest, GetRandomMoveIndexTest)
{
    size_t randomResult = 0;
    CudaUtils::Memory<size_t> cudaRandomResult(1);
    cudaRandomResult.transferFrom(&randomResult);
    vector<size_t> limits = {2, 7, 11, 12, 15, 21, 25, 42, 64};
    for (const auto& limit : limits)    
    {
        for (size_t i = 1; i <= limit; ++i)
        {
            size_t result = limit - i;
            testRandomNumberProxy(result * 1.0f / limit, limit, cudaRandomResult.get());

            cudaRandomResult.transferTo(&randomResult);

            ASSERT_EQ(result, randomResult)<<" for limit "<<limit<<" ratio: "<<result * 1.0f / limit;
        }
    }
}

TEST_F(OthelloCudaUtilTest, GetRandomMoveIndexRegressionTest)
{
    size_t randomResult = 0;
    CudaUtils::Memory<size_t> cudaRandomResult(1);
    cudaRandomResult.transferFrom(&randomResult);
    // fakeRandom, maximum, expected result
    typedef tuple<float, size_t, size_t> scenario; 
    vector<scenario> data = { // the commented results are expected but not calculated due to rounding issues
        scenario(0.590909,            22, 12),// 13),
        scenario(0.979073,             4,  3),
        scenario(0.69767439365386963, 43, 29),// 30),
        scenario(0.7804877758026123,  41, 31), // 32)
        scenario(1.0,                 10,  9)
    }; 
    for (const auto& t : data)
    {
        testRandomNumberProxy(get<0>(t), get<1>(t), cudaRandomResult.get());

        cudaRandomResult.transferTo(&randomResult);

        ASSERT_EQ(get<2>(t), randomResult);
    }
}

TEST_F(OthelloCudaUtilTest, CalculateNumberOfMarkedFieldsAllTrueTest)
{
    size_t sumResult = 0;
    CudaUtils::Memory<size_t> cudaSumResult(1);
    cudaSumResult.transferFrom(&sumResult);

    bool playfield[64] = {
        true, true, true, true, true, true, true, true, 
        true, true, true, true, true, true, true, true, 
        true, true, true, true, true, true, true, true, 
        true, true, true, true, true, true, true, true, 
        true, true, true, true, true, true, true, true, 
        true, true, true, true, true, true, true, true, 
        true, true, true, true, true, true, true, true, 
        true, true, true, true, true, true, true, true, 
    };
    CudaUtils::Memory<bool> cudaPlayfield(64);
    cudaPlayfield.transferFrom(playfield);

    testNumberOfMarkedFieldsProxy(cudaSumResult.get(), cudaPlayfield.get());

    cudaSumResult.transferTo(&sumResult);
    ASSERT_EQ(64u, sumResult);
}

TEST_F(OthelloCudaUtilTest, CalculateNumberOfMarkedFieldsAllFalseTest)
{
    size_t sumResult = 0;
    CudaUtils::Memory<size_t> cudaSumResult(1);
    cudaSumResult.transferFrom(&sumResult);

    bool playfield[64] = {
        false, false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, false, 
    };
    CudaUtils::Memory<bool> cudaPlayfield(64);
    cudaPlayfield.transferFrom(playfield);

    testNumberOfMarkedFieldsProxy(cudaSumResult.get(), cudaPlayfield.get());

    cudaSumResult.transferTo(&sumResult);
    ASSERT_EQ(0u, sumResult);
}

TEST_F(OthelloCudaUtilTest, CalculateNumberOfMarkedFieldsSomeTrueTest)
{
    size_t sumResult = 0;
    CudaUtils::Memory<size_t> cudaSumResult(1);
    cudaSumResult.transferFrom(&sumResult);

    bool playfield[64] = {
        false, true,  false, false, false, false, true,  false, 
        false, false, false, false, false, true,  false, false, 
        false, false, false, false, true,  false, false, false, 
        true,  false, false, true,  false, false, false, true, 
        false, false, true,  false, false, false, false, false, 
        false, true,  false, false, false, false, false, false, 
        true,  false, false, false, false, false, false, false, 
        false, false, false, false, false, false, false, true, 
    };
    CudaUtils::Memory<bool> cudaPlayfield(64);
    cudaPlayfield.transferFrom(playfield);

    testNumberOfMarkedFieldsProxy(cudaSumResult.get(), cudaPlayfield.get());

    cudaSumResult.transferTo(&sumResult);
    ASSERT_EQ(11u, sumResult);
}

TEST_F(OthelloCudaUtilTest, CalculateNumberOfMarkedFieldsManyTrueTest)
{
    size_t sumResult = 0;
    CudaUtils::Memory<size_t> cudaSumResult(1);
    cudaSumResult.transferFrom(&sumResult);

    bool playfield[64] = {
        true,  false, true,  true,  true,  true,  false, true, 
        false, true,  true,  false, true,  true,  true,  false, 
        true,  true,  true,  true,  true,  false, true,  true, 
        true,  true,  true,  true,  false, true,  true,  true, 
        false, true,  true,  true,  true,  true,  true,  true, 
        true,  true,  false, true,  true,  true,  true,  true, 
        true,  true,  true,  true,  true,  true,  true,  false, 
        true,  false, true,  true,  true,  true,  true,  true, 
    };
    CudaUtils::Memory<bool> cudaPlayfield(64);
    cudaPlayfield.transferFrom(playfield);

    testNumberOfMarkedFieldsProxy(cudaSumResult.get(), cudaPlayfield.get());

    cudaSumResult.transferTo(&sumResult);
    ASSERT_EQ(53u, sumResult);
}

TEST_F(OthelloCudaUtilTest, CalculateNumberOfMarkedFieldsTrueAndFalseEqualTest)
{
    size_t sumResult = 0;
    CudaUtils::Memory<size_t> cudaSumResult(1);
    cudaSumResult.transferFrom(&sumResult);

    bool playfield[64] = {
        false, false, true,  true,  true,  true,  false, false, 
        false, false, true,  false, true,  false, false, false, 
        false, false, false, false, false, false, true,  true, 
        false, false, false, false, false, true,  true,  true, 
        false, true,  false, true,  false, true,  false, true, 
        true,  false, false, true,  false, true,  true,  true, 
        true,  true,  true,  false, true,  true,  true,  false, 
        true,  false, true,  false, true,  true,  true,  true, 
    };
    CudaUtils::Memory<bool> cudaPlayfield(64);
    cudaPlayfield.transferFrom(playfield);

    testNumberOfMarkedFieldsProxy(cudaSumResult.get(), cudaPlayfield.get());

    cudaSumResult.transferTo(&sumResult);
    ASSERT_EQ(32u, sumResult);
}