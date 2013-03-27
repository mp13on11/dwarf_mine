#pragma once

#include <OthelloField.h>
#include <OthelloResult.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

extern void gameSimulation(size_t numberOfBlocks, size_t iterations, size_t* seeds, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, OthelloResult* results);
extern void gameSimulationPreRandom(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, OthelloResult* results);
extern void gameSimulationPreRandom(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, OthelloResult* results);
extern void gameSimulationStreamed(size_t numberOfBlocks, size_t iterations, size_t* seeds, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, OthelloResult* results, cudaStream_t stream);

extern void testNumberOfMarkedFieldsProxy(size_t* sum, const bool* playfield);
extern void testRandomNumberProxy(float fakedRandom, size_t maximum, size_t* randomMoveIndex);
extern void testDoStepProxy(Field* playfield, Player currentPlayer, float fakedRandom);
extern void testExpandLeafProxy(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);
