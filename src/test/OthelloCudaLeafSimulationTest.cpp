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

#include "OthelloCudaLeafSimulationTest.h"
#include "State.h"
#include "Move.h"
#include "OthelloUtil.h"
#include <cuda-utils/Memory.h>
#include <functional>
#include <vector>
#include <utility>
#include <stdexcept>


using namespace std;

#define ASSERT_EQ_VECTOR(ACTUAL, EXPECTED) \
    ASSERT_EQ(ACTUAL.size(), EXPECTED.size()); \
    for (size_t i = 0; i < ACTUAL.size(); ++i) \
        ASSERT_EQ(ACTUAL[i], EXPECTED[i])<<"\t  at index: "<<i;

Playfield getExpectedPlayfield(Playfield& playfield, vector<pair<size_t, Field>> expectedChanges)
{
    Playfield expectedPlayfield;
    expectedPlayfield.assign(playfield.begin(), playfield.end());
    for (const auto change : expectedChanges)
    {
        expectedPlayfield[change.first] = change.second;
    }
    return expectedPlayfield;
}

void testSingleStep(Playfield& playfield, vector<pair<size_t, Field>> expectedChanges, Player currentPlayer, float randomFake)
{
    CudaUtils::Memory<Field> cudaPlayfield(playfield.size());
    cudaPlayfield.transferFrom(playfield.data());
    
    testDoStepProxy(cudaPlayfield.get(), currentPlayer, randomFake);

    Playfield outputPlayfield(playfield.size());
    cudaPlayfield.transferTo(outputPlayfield.data());

    Playfield expectedPlayfield = getExpectedPlayfield(playfield, expectedChanges);

    // State temp(outputPlayfield, Player::White);
    // cout << "Actual: \n"<<temp << endl;

    // State temp2(expectedPlayfield, Player::White);
    // cout << "Expected: \n"<<temp2 << endl;

    ASSERT_EQ_VECTOR(outputPlayfield, expectedPlayfield);
}

void testMultipleSteps(Playfield& playfield, vector<pair<size_t, Field>> expectedChanges, Player currentPlayer, size_t expectedWins, size_t expectedVisits )
{
    CudaUtils::Memory<Field> cudaPlayfield(playfield.size());
    cudaPlayfield.transferFrom(playfield.data());
    
    CudaUtils::Memory<size_t> cudaWins(1);
    CudaUtils::Memory<size_t> cudaVisits(1);
    size_t wins = 0;
    size_t visits = 0;

    cudaWins.transferFrom(&wins);
    cudaVisits.transferFrom(&visits);

    testExpandLeafProxy(cudaPlayfield.get(), currentPlayer, cudaWins.get(), cudaVisits.get());

    cudaWins.transferTo(&wins);
    cudaVisits.transferTo(&visits);


    Playfield outputPlayfield(playfield.size());
    cudaPlayfield.transferTo(outputPlayfield.data());

    Playfield expectedPlayfield = getExpectedPlayfield(playfield, expectedChanges);

    // State temp(outputPlayfield, Player::White);
    // cout << "Actual: \n"<<temp << endl;

    // State temp2(expectedPlayfield, Player::White);
    // cout << "Expected: \n"<<temp2 << endl;

    ASSERT_EQ(expectedVisits, visits);
    ASSERT_EQ(expectedWins, wins);
    ASSERT_EQ_VECTOR(outputPlayfield, expectedPlayfield);
}

Playfield singleMoveSingleFlipPlayfield { 
    F, F, F, F, F, F, F, F, 
    F, F, F, F, F, F, F, F, 
    F, F, F, F, F, F, F, F, 
    F, F, F, W, B, F, F, F, 
    F, F, F, B, W, F, F, F, 
    F, F, F, F, F, F, F, F, 
    F, F, F, F, F, F, F, F, 
    F, F, F, F, F, F, F, F
};

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveSingleFlipTest1)
{
    testSingleStep(singleMoveSingleFlipPlayfield, {{20, W}, {28, W}}, W, 0 * 1.0 / 4);
}

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveSingleFlipTest2)
{
    testSingleStep(singleMoveSingleFlipPlayfield, {{28, W}, {29, W}}, W, 1 * 1.0 / 4);
}

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveSingleFlipTest3)
{
    testSingleStep(singleMoveSingleFlipPlayfield, {{34, W}, {35, W}}, W, 2 * 1.0 / 4);
}

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveSingleFlipTest4)
{
    testSingleStep(singleMoveSingleFlipPlayfield, {{35, W}, {43, W}}, W, 3 * 1.0 / 4);
}

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveSingleFlipTest5)
{
    testSingleStep(singleMoveSingleFlipPlayfield, {{26, B}, {27, B}}, B, 0.25);   
}

Playfield singleMoveMultipleFlipPlayfield {
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, W, W, F, B, F, F, F, 
        B, W, B, W, W, W, W, F, 
        W, B, B, B, B, B, F, F, 
        W, W, W, W, W, W, F, F, 
        F, F, B, W, W, F, F, F, 
        F, F, B, F, F, F, F, F
    };

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveMultipleFlipTest1)
{
    testSingleStep(singleMoveMultipleFlipPlayfield, {{48, B}, {32, B}, {40, B}, {41, B}}, B,  10 * 1.0 / 17);   
}

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveMultipleFlipTest2)
{
    testSingleStep(singleMoveMultipleFlipPlayfield, {{53, B}, {51, B}, {52, B}, {44, B}, {45, B}}, B, 12 * 1.0 / 17);   
}


TEST_F(OthelloCudaLeafSimulationTest, NoPossibleSingleMoveTest)
{
    Playfield playfield {
        W, W, W, W, W, W, W, W, 
        W, W, F, F, F, F, F, F, 
        W, W, F, B, B, B, B, B, 
        W, W, F, B, B, B, B, B, 
        W, W, F, B, B, B, B, B, 
        W, W, F, B, B, B, B, B, 
        W, W, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F
    };

    testSingleStep(playfield, {}, W,  0 * 1.0 / 1);   
    testSingleStep(playfield, {}, B,  0 * 1.0 / 1);   
}


TEST_F(OthelloCudaLeafSimulationTest, NoPossibleMovesTest)
{
    Playfield playfield {
        W, W, W, W, W, W, W, W, 
        W, W, F, F, F, F, F, F, 
        W, W, F, B, B, B, B, B, 
        W, W, F, B, B, B, B, B, 
        W, W, F, B, B, B, B, B, 
        W, W, F, B, B, B, B, B, 
        W, W, F, F, F, F, F, F, 
        W, W, F, F, F, F, F, F
    };
    // 22 W
    // 20 B

    testMultipleSteps(playfield, {}, W, 1, 1);
    testMultipleSteps(playfield, {}, B, 0, 1);  
}

TEST_F(OthelloCudaLeafSimulationTest, PredictableMoveTest)
{
    Playfield playfield {
        F, F, F, F, W, F, F, F, 
        W, F, F, F, W, F, F, F, 
        F, W, F, F, W, F, F, W, 
        F, F, W, F, W, F, W, F, 
        F, F, F, W, W, W, F, F, 
        W, W, W, W, B, W, W, W, 
        F, F, F, W, W, W, F, F, 
        F, F, W, W, F, F, W, F
    };
    testSingleStep(playfield, {{60, B}, {52, B}}, B,  0);   
}


TEST_F(OthelloCudaLeafSimulationTest, PredictableFinalMoveTest)
{
    Playfield playfield {
        F, F, F, F, W, F, F, F, 
        W, F, F, F, W, F, F, F, 
        F, W, F, F, W, F, F, W, 
        F, F, W, F, W, F, W, F, 
        F, F, F, W, W, W, F, F, 
        W, W, W, W, B, W, W, W, 
        F, F, F, W, B, W, F, F, 
        F, F, W, W, B, F, W, F
    };
    testSingleStep(playfield, {{60, W}, {61, W}, {52, W}}, W,  0);   
}

TEST_F(OthelloCudaLeafSimulationTest, PredictableLoserGameTest)
{
    Playfield playfield {
        F, F, F, F, W, F, F, F, 
        W, F, F, F, W, F, F, F, 
        F, W, F, F, W, F, F, W, 
        F, F, W, F, W, F, W, F, 
        F, F, F, W, W, W, F, F, 
        W, W, W, W, B, W, W, W, 
        F, F, F, W, W, W, F, F, 
        F, F, W, W, F, F, W, F
    };

    testMultipleSteps(playfield, {{60, W}, {61, W}}, B, 0, 1);
}

TEST_F(OthelloCudaLeafSimulationTest, PredictableWinnerGameTest)
{
    Playfield playfield {
        F, F, F, F, W, F, F, F, 
        W, F, F, F, W, F, F, F, 
        F, W, F, F, W, F, F, W, 
        F, F, W, F, W, F, W, F, 
        F, F, F, W, W, W, F, F, 
        W, W, W, W, B, W, W, W, 
        F, F, F, W, B, W, F, F, 
        F, F, W, W, B, F, W, F
    };

    testMultipleSteps(playfield, {{52, W}, {60, W}, {61, W}}, W, 1, 1);
}

TEST_F(OthelloCudaLeafSimulationTest, BoundarySingleStepSingleFlipTest)
{
    Playfield playfield {
        F, F, F, F, F, F, F, F, 
        F, F, F, F, B, F, F, F, 
        F, F, F, F, B, F, F, F, 
        F, F, F, B, B, F, F, F, 
        F, W, W, W, B, B, F, F, 
        F, B, F, F, W, F, B, F, 
        F, F, F, F, F, W, F, F, 
        F, F, F, F, F, F, F, F
    };
    // possible: 25, 32, 42, 43, 51, 52, 60
    testSingleStep(playfield, {{53, B}, {60, B}}, B,  6 * 1.0 / 7);   
}
