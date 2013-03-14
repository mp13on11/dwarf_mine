#pragma once

#define OTHELLO_DEBUG

#ifdef OTHELLO_DEBUG
    #include <cassert>
    #include <cstdio>

    const int MAXIMAL_MOVE_COUNT = 128; // an impossible condition - it would mean that for every field both players had to pass

    #define cassert(CONDITION, MESSAGE, ...) \
        if (!(CONDITION)) \
        {   \
            printf(MESSAGE, __VA_ARGS__); \
            assert(CONDITION); \
        }
    #pragma message "\n=====================\n\n== CUDA DEBUG MODE ==\n\n====================="
#else
    #define cassert(CONDITION, MESSAGE, ...)
#endif