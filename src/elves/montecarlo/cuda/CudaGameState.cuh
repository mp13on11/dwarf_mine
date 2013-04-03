#pragma once

#include "OthelloField.h"
#include "CudaUtil.cuh"

typedef struct _CudaGameState
{
    Field* field;
    Field* oldField;
    bool* possible;
    size_t size;
    size_t sideLength;
    Player currentPlayer;

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

    __device__ bool isWinner(Player requestedPlayer)
    {
        Player enemyPlayer = getEnemyPlayer(requestedPlayer);
/*
        __shared__ unsigned int enemyCounter[FIELD_DIMENSION];
        __shared__ unsigned int requestedCounter[FIELD_DIMENSION];
        int idMod = threadIdx.x % FIELD_DIMENSION;
        int idDiv = threadIdx.x / FIELD_DIMENSION;
        if (idMod == 0) enemyCounter[idDiv] = requestedCounter[idDiv] = 0;

        __syncthreads();

        if (field[threadIdx.x] == enemyPlayer) atomicAdd(&enemyCounter[idDiv], 1u);
        if (field[threadIdx.x] == requestedPlayer) atomicAdd(&requestedCounter[idDiv], 1u);

        __syncthreads();

        if (idMod == 0 && threadIdx.x != 0) 
        {
            atomicAdd(&enemyCounter[0], enemyCounter[idDiv]);
            atomicAdd(&requestedCounter[0], requestedCounter[idDiv]);
        }
*/

        __shared__ unsigned int enemyCounter[8];
        __shared__ unsigned int requestedCounter[8];

        if (threadIdx.x % 8 == 0) enemyCounter[threadIdx.x / 8] = requestedCounter[threadIdx.x / 8] = 0;

        __syncthreads();

        if (field[threadIdx.x] == enemyPlayer) atomicAdd(&enemyCounter[threadIdx.x / 8], 1u);
        if (field[threadIdx.x] == requestedPlayer) atomicAdd(&requestedCounter[threadIdx.x / 8], 1u);

        __syncthreads();

        if (threadIdx.x % 8 == 0 && threadIdx.x != 0) 
        {
            atomicAdd(&enemyCounter[0], enemyCounter[threadIdx.x / 8]);
            atomicAdd(&requestedCounter[0], requestedCounter[threadIdx.x / 8]);
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