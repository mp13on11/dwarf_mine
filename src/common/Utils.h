#pragma once


template<typename NumberType>
inline NumberType div_ceil(const NumberType& a, const NumberType& b)
{
    return 1 + ((a - 1) / b);
}
