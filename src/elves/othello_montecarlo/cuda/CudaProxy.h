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

#include <Field.h>
#include <Result.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

extern void gameSimulationPreRandomStreamed(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results, cudaStream_t stream, size_t streamSeed);
extern void gameSimulationPreRandom(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results);


extern void testNumberOfMarkedFieldsProxy(size_t* sum, const bool* playfield);
extern void testRandomNumberProxy(float fakedRandom, size_t maximum, size_t* randomMoveIndex);
extern void testDoStepProxy(Field* playfield, Player currentPlayer, float fakedRandom);
