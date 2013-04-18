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

#include "Factorize.h"
#include <cuda.h>
#include <stdio.h>

#define USE_LONG_DIVISION

struct Number
{
    NumData fields;

    static __device__ Number ZERO()
    {
        return Number(static_cast<uint64_t>(0));
    }

    __device__ Number() 
    {
        memset(fields, 0, DATA_SIZE_BYTES);
    }

     __device__ Number(const uint64_t data)
     {
         fields[0] = static_cast<uint32_t>(data);
         fields[1] = static_cast<uint32_t>(data >> 32);
         memset(fields + 2, 0, (NUM_FIELDS - 2)*4);
     }

     __device__ int bitAt(int index)
     {
         int field = index / 32; 
         int index2 = index % 32;

         return (fields[field] >> index2) & 1; 
     }

    __device__ Number(const NumData data)
    {
        memcpy(fields, data, DATA_SIZE_BYTES);
    }

    __device__ void writeTo(PNumData out)
    {
        memcpy(out, fields, DATA_SIZE_BYTES);
    }

    __device__ uint32_t get_ui() const
    {
        return fields[0];
    }

    __device__ uint64_t get_ui64() const
    {
        return fields[0] | ((uint64_t)fields[1] << 32);
    }

    __device__ bool isZero() const 
    {
        for (int i = 0; i < NUM_FIELDS; ++i)
        {
            if (fields[i] != 0)
            {
                return false;
            }
        }
        return true;
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
        return *this;
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
        memset(nfields, 0, DATA_SIZE_BYTES);

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

        memcpy(fields, nfields, DATA_SIZE_BYTES);

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
        Number remainder(*this);
        Number quotient = remainder.divMod(other);

        memcpy(fields, quotient.fields, DATA_SIZE_BYTES);
        return *this;
    }

    __device__ Number operator%(const Number& other) const
    {
        Number remainder(*this);
        remainder.divMod(other);
        return remainder;
    }

    __device__ Number divMod(const Number& other)
    {
#ifdef USE_LONG_DIVISION
        if(this == &other)
        {
            *this = Number(static_cast<uint64_t>(0));
            return Number(1);
        }

        Number quotient(static_cast<uint64_t>(0));
        Number rest(static_cast<uint64_t>(0));

        for (int i = NUM_FIELDS - 1; i >= 0; i--)
        {
            rest <<= 32;
            rest.fields[0] = this->fields[i];

            while (rest >= other)
            {
                rest -= other;
                quotient.fields[i] += (uint32_t) 1;
            }
        }
        *this = rest;
        return quotient;

#else
      if(this == &other)
      {
          *this = Number(static_cast<uint64_t>(0));
          return Number(1);
      }

      Number quotient(static_cast<uint64_t>(0));

      if (*this < other)
      {
         return Number::ZERO();
      }

      while (*this >= other)
      {
          *this = *this - other;
          quotient += 1;
      }
      return quotient;
#endif
    }

    __device__ Number operator/(const Number& other) const
    {
        Number result(*this);
        result /= other;
        return result;
    }

    __device__ bool operator<(const Number& other) const
    {
        for (int i = NUM_FIELDS-1; i >= 0; --i)
        {
            if (fields[i] == other.fields[i]) continue;
            else if (fields[i] < other.fields[i])
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        return false;
    }

    __device__ bool operator<=(const Number& other) const
    {
        return *this < other || *this == other;
    }

    __device__ bool operator>(const Number& other) const
    {
        return !(*this < other) && !(*this == other);
    }

    __device__ bool operator>=(const Number& other) const
    {
        return (other < *this) || (*this == other);
    }

    __device__ bool operator==(const Number& other) const
    {
       return !(*this < other) && !(other < *this);
    }

    __device__ bool operator !=(const Number& other) const
    {
        return !(*this == other);
    }

    __device__ Number& operator<<=(uint32_t offset)
    {
        uint32_t itemOffset = offset % 32;
        uint32_t blockOffset = offset / 32;
        uint32_t carry = 0;

        blockOffset = blockOffset > NUM_FIELDS ? NUM_FIELDS: blockOffset;

        for (int i = 0; i < NUM_FIELDS; ++i)
        {
            uint32_t old = fields[i];
            fields[i] = carry | (fields[i] << itemOffset);
            carry = old >> (32 - itemOffset);
        }

        for (int i = NUM_FIELDS-1; blockOffset > 0 && i >= blockOffset; --i)
        {
            fields[i] = fields[i - blockOffset];
        }
        if (blockOffset > 0)
            memset(fields, 0, blockOffset);

        return *this;
    }

    __device__ Number operator<<(uint32_t offset) const
    {
        Number result(*this);
        result <<= offset;
        return result;
    }

    __device__ Number& operator>>=(uint32_t offset)
    {
        uint32_t itemOffset = offset % 32;
        uint32_t blockOffset = offset / 32;
        uint32_t carry = 0;

        blockOffset = blockOffset > NUM_FIELDS ? NUM_FIELDS : blockOffset;

        for (int i = NUM_FIELDS-1; i >= 0 ; --i)
        {
            if (fields[i] == 0) continue;

            uint32_t old = fields[i];
            fields[i] = carry | (fields[i] >> itemOffset);
            carry = old << (32 - itemOffset);
        }

        for (int i = 0; i < NUM_FIELDS - blockOffset; ++i)
        {
            fields[i] = fields[i + blockOffset];
        }

        for (int i = NUM_FIELDS - blockOffset;  i < NUM_FIELDS; ++i)
        {
            fields[i] = 0;
        }

        return *this;
    }

    __device__ Number operator>>(uint32_t offset) const
    {
        Number result(*this);
        result >>= offset;
        return result;
    }

};


