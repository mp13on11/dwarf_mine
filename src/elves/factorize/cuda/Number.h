#pragma once

#include "Factorize.h"
#include <cuda.h>
#include <stdio.h>

struct Number
{
    NumData fields;

    __device__ Number(const NumData data)
    {
        memcpy(fields, data, sizeof(uint32_t)*NUM_FIELDS);
    }

    __device__ Number& operator+=(const Number& other)
    {
        uint32_t carry = 0;
        for (int i = 0; i < NUM_FIELDS; ++i)
        {
            uint64_t result = static_cast<uint64_t>(fields[i]) + other.fields[i] + carry;
            this->fields[i] = static_cast<uint32_t>(result);
            carry = result >> 32;

        }
        return *this;
    }

    __device__ Number operator+(const Number& other) const
    {
        Number result(*this);
        result += other;
        return result;
    }

    __device__ Number& operator-=(const Number& other)
    {
        uint64_t carry = 0;
        for (int i = 0; i < NUM_FIELDS; ++i)
        {
            uint64_t result = static_cast<uint64_t>(fields[i]) - other.fields[i] - carry;
            this->fields[i] = static_cast<uint32_t>(result);
            carry = result >> 63;
        }
    }

    __device__ Number operator-(const Number& other) const
    {
        Number result(*this);
        result -= other;
        return result;
    }

    __device__ Number& operator*=(const Number& other)
    {
        for () {
        }
    }

    __device__ Number operator*(const Number& other) const
    {
        Number result(*this);
        result *= other;
        return result;
    }

    __device__ Number& operator/=(const Number& other)
    {

    }

    __device__ Number operator/(const Number& other) const
    {
        Number result(*this);
        result /= other;
        return result;
    }
};
