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

#include "Field.h"
#include "Random.cuh"

typedef struct _State
{
    Field* field;
    Field* oldField;
    bool* possible;
    size_t size;
    size_t sideLength;
    Player currentPlayer;

    __device__ _State(Field* field, Field* oldField, bool* possibleMoves, size_t fieldDimension, Player currentPlayer)
        : field(field), oldField(oldField), possible(possibleMoves), size(fieldDimension * fieldDimension), sideLength(fieldDimension), currentPlayer(currentPlayer)
    {

    }

    __device__ inline bool inBounds(int x, int y)
    {
        return (x >= 0 && x < sideLength && y >= 0 && y < sideLength);
    }

    __device__ inline bool inBounds(int i)
    {
        return (i >= 0 && i < size);
    }

    __device__ inline Player getEnemyPlayer(Player player)
    {
        return (player == Black ? White : Black);
    }

    __device__ inline Player getEnemyPlayer()
    {
        return getEnemyPlayer(currentPlayer);
    }

    __device__ inline void switchPlayer()
    {
        currentPlayer = getEnemyPlayer();
    }

    __device__ inline size_t numberOfMarkedFields()
    {
        __shared__ unsigned int s[8];
        if (threadIdx.x % 8 == 0) s[threadIdx.x / 8] = 0;
        __syncthreads();
        if (possible[threadIdx.x]) atomicAdd(&s[threadIdx.x / 8], 1u);
        __syncthreads();
        if (threadIdx.x % 8 == 0 && threadIdx.x != 0) atomicAdd(&s[0], s[threadIdx.x / 8]);
        __syncthreads();
        return s[0];
    }

    __device__ bool isWinner(Player requestedPlayer)
    {
        Player enemyPlayer = getEnemyPlayer(requestedPlayer);
        __shared__ unsigned int enemyCounter[FIELD_DIMENSION];
        __shared__ unsigned int requestedCounter[FIELD_DIMENSION];

        if (threadIdx.x % FIELD_DIMENSION == 0) enemyCounter[threadIdx.x / FIELD_DIMENSION] = requestedCounter[threadIdx.x / 8] = 0;

        __syncthreads();

        if (field[threadIdx.x] == enemyPlayer) atomicAdd(&enemyCounter[threadIdx.x / FIELD_DIMENSION], 1u);
        if (field[threadIdx.x] == requestedPlayer) atomicAdd(&requestedCounter[threadIdx.x / 8], 1u);

        __syncthreads();

        if (threadIdx.x % FIELD_DIMENSION == 0 && threadIdx.x != 0) 
        {
            atomicAdd(&enemyCounter[0], enemyCounter[threadIdx.x / FIELD_DIMENSION]);
            atomicAdd(&requestedCounter[0], requestedCounter[threadIdx.x / FIELD_DIMENSION]);
        }

        __syncthreads();
    
        return requestedCounter[0] > enemyCounter[0];
    }

    __device__ bool isUnchanged()
    {
        __syncthreads();
        bool same = true;
        for (size_t i = 0; i < size; i++)
        {
            same &= (oldField[i] == field[i]);
        }
        return same;
    }
} CudaGameState;