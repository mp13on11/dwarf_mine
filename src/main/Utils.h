#pragma once

inline size_t div_ceil(size_t x, size_t y)
{
    return (x % y) ? (x / y + 1) : (x / y);
}
