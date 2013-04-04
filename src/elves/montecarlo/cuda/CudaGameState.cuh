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

    #define ID_MOD threadIdx.x % 8
    #define ID_DIV threadIdx.x / 8

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