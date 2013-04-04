#pragma once

#include <Field.h>
#include <Result.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

extern void gameSimulationPreRandomStreamed(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results, cudaStream_t stream, size_t streamSeed);
extern void gameSimulationPreRandom(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results, cudaStream_t stream);
extern void gameSimulationPreRandom(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results);


extern void testNumberOfMarkedFieldsProxy(size_t* sum, const bool* playfield);
extern void testRandomNumberProxy(float fakedRandom, size_t maximum, size_t* randomMoveIndex);
extern void testDoStepProxy(Field* playfield, Player currentPlayer, float fakedRandom);
extern void testExpandLeafProxy(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);
