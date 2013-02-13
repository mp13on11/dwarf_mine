#pragma once



typedef struct _CudaGameState
{
    Field* field;
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

    __device__ inline Player getEnemyPlayer(Player currentPlayer)
    {
        return (currentPlayer == Black ? White : Black);
    }

    __device__ inline Player getEnemyPlayer()
    {
        return getEnemyPlayer(currentPlayer);
    }

    __device__ bool isWinner(Player requestedPlayer)
    {
        Player enemyPlayer = getEnemyPlayer(requestedPlayer);
        size_t requestedCounters = 0;
        size_t enemyCounters = 0;
        for (size_t i = 0; i < size; ++i)
        {
            if (field[i] == enemyPlayer) ++enemyCounters;
            if (field[i] == requestedPlayer) ++requestedCounters;
        }
        return requestedPlayer >= enemyPlayer;
    }
} CudaGameState;