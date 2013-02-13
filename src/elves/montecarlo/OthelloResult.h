#pragma once

struct OthelloResult
{
    size_t x;
    size_t y;
    size_t visits;
    size_t wins;
    size_t iterations;

    bool equalMove(const OthelloResult& other) const
    {
        return (x == other.x && y == other.y);
    }

    double successRate() const
    {
        return 1.0 * wins / visits;
    }
};