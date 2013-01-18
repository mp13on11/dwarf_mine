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
        NumData nfields;
        memset(nfields, 0, sizeof(uint32_t) * NUM_FIELDS);

        uint32_t carry = 0;

        for (int r = 0; r < NUM_FIELDS/2; ++r)
        {

            uint32_t low = 0;
            uint32_t high = 0;

            for (int l = 0; l < NUM_FIELDS/2; ++l)
            {
                uint64_t mulResult = fields[l] * static_cast<uint64_t>(other.fields[r]);

                low = static_cast<uint32_t>(mulResult);

                uint64_t addResult = nfields[r+l] + static_cast<uint64_t>(low) + high;
                nfields[r+l] = static_cast<uint32_t>(addResult) + carry;
                carry = static_cast<uint32_t>(addResult >> 32);

                high = static_cast<uint32_t>(mulResult >> 32);

            }
        }

        memcpy(fields, nfields, sizeof(uint32_t)*NUM_FIELDS);

        return *this;
    }

    __device__ Number operator*(const Number& other) const
    {
        Number result(*this);
        result *= other;
        return result;
    }

    __device__ Number& operator/=(const Number& other)
    {
        //Number result(1);


    }

    __device__ Number operator/(const Number& other) const
    {
        Number result(*this);
        result /= other;
        return result;
    }
};
