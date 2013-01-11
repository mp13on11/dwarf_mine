#include "BigInt.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <istream>
#include <sstream>
#include <stdexcept>

using namespace std;

const BigInt& BigInt::ZERO = BigInt(0);
const BigInt& BigInt::ONE = BigInt(1);

BigInt::BigInt()
{
    items.push_back(0);
}

BigInt::BigInt(uint32_t value)
{
    items.push_back(value);
}

BigInt::BigInt(const std::vector<uint32_t>& items) :
        items(items)
{
    normalize();
}

BigInt::BigInt(const string& value)
{
    istringstream stream(value);
    readFrom(stream);

    if (stream.fail())
        throw logic_error("Failed to convert string to BigInt: " + value);
}

BigInt& BigInt::operator=(uint32_t value)
{
    items.clear();
    items.push_back(value);

    return *this;
}

BigInt& BigInt::operator=(const string& value)
{
    istringstream stream(value);
    readFrom(stream);

    if (stream.fail())
        throw logic_error("Failed to convert string to BigInt: " + value);

    return *this;
}

BigInt& BigInt::operator+=(const BigInt& right)
{
    if(this == &right)
    {
        return *this <<= 1;
    }

    size_t carry = 0;
    size_t minimum = min(items.size(), right.items.size());
    size_t maximum = max(items.size(), right.items.size());

    for (size_t i=0; i<minimum; i++)
    {
        uint64_t res = (uint64_t)items[i] + (uint64_t)right.items[i] + (uint64_t)carry;
        items[i] = res;
        carry = res >> 32;
    }

    if (items.size() < right.items.size())
    {
        items.insert(
                items.end(),
                right.items.begin() + items.size(), right.items.end()
            );
    }

    for (size_t i=minimum; i<maximum && carry > 0; i++)
    {
        uint64_t res = (uint64_t)items[i] + (uint64_t)carry;
        items[i] = res;
        carry = res >> 32;
    }

    if (carry == 1)
    {
        items.push_back(carry);
    }

    return *this;
}

BigInt& BigInt::operator-=(const BigInt& right)
{    
    if(this == &right)
    {
        return *this = BigInt::ZERO;
    }

    if (*this < right)
        throw logic_error("underflow during subtraction");

    uint32_t carry = 0;
    for (size_t i=0; i<right.items.size(); i++)
    {
        uint32_t old = items[i];
        items[i] -= right.items[i] + carry;
        carry = 0;

        if (items[i] > old) // underflow
        {
            carry = 1;
        }
    }

    for (size_t i=right.items.size(); i<items.size() && carry > 0; i++)
    {
        uint32_t old = items[i];
        items[i] -= carry;
        carry = 0;

        if (items[i] > old) // underflow
        {
            carry = 1;
        }
    }

    normalize();

    return *this;
}

#include <iostream>
BigInt& BigInt::operator*=(const BigInt& factor)
{
    if(this == &factor)
    {
        return *this *= BigInt(factor);
    }

    BigInt result(0);

    for(size_t i=0; i<factor.items.size(); i++)
    {
        BigInt tempResult(0);
        tempResult.items.clear();
        uint32_t carry = 0;
        uint64_t smallFactor = factor.items[i];
        if(smallFactor == 0)
            continue;

        for(size_t k=0; k<this->items.size(); k++)
        {
            uint64_t res = this->items[k] * smallFactor + carry;
            tempResult.items.push_back(res);
            carry = res >> 32;
        }
        if(carry > 0)
            tempResult.items.push_back(carry);

        tempResult <<= 32 * i;
        result += tempResult;
    }

    *this = result;
    return *this;
}

BigInt& BigInt::operator/=(const BigInt& right)
{
    *this = divMod(right);

    return *this;
}

BigInt& BigInt::operator%=(const BigInt& right)
{
    divMod(right);

    return *this;
}

BigInt& BigInt::operator<<=(uint32_t offset)
{
    uint32_t itemOffset = offset % BITS_PER_ITEM;
    uint32_t blockOffset = offset / BITS_PER_ITEM;
    uint32_t carry = 0;

    if (itemOffset != 0)
    {
        for (size_t i=0; i<items.size(); ++i)
        {
            uint32_t old = items[i];
            items[i] = carry | (old << itemOffset);
            carry = old >> (BITS_PER_ITEM - itemOffset);
        }
    }

    if (carry > 0)
    {
        items.push_back(carry);
    }

    items.insert(items.begin(), blockOffset, 0);

    this->normalize();
    return *this;
}

BigInt& BigInt::operator>>=(uint32_t offset)
{
    uint32_t itemOffset = offset % BITS_PER_ITEM;
    uint32_t blockOffset = offset / BITS_PER_ITEM;
    uint32_t carry = 0;

    items.erase(items.begin(), items.begin() + blockOffset);

    if (itemOffset > 0)
    {
        for (int i=items.size() - 1; i >= 0; --i)
        {
            uint32_t old = items[i];
            items[i] = carry | (old >> itemOffset);
            carry = old << (BITS_PER_ITEM - itemOffset);
        }
    }

    normalize();

    return *this;
}

bool BigInt::operator==(const BigInt& right) const
{
    if (items.size() != right.items.size())
        return false;

    for (size_t i=0; i<items.size(); ++i)
    {
        if (items[i] != right.items[i])
            return false;
    }

    return true;
}

bool BigInt::operator>=(const BigInt& right) const
{
    if (items.size() < right.items.size())
        return false;
    if (items.size() > right.items.size())
        return true;

    for (int i=items.size() - 1; i >= 0; --i)
    {
        if (items[i] < right.items[i])
            return false;
        if (items[i] > right.items[i])
            return true;
    }

    // both are equal
    return true;
}

string BigInt::toString() const
{
    vector<uint32_t> printableItems;

    BigInt remainder(*this);
    BigInt maximumPrintableItem(MAXIMUM_PRINTABLE_ITEM);

    while (remainder > maximumPrintableItem)
    {
        BigInt quotient = remainder.divMod(maximumPrintableItem);
        printableItems.push_back(remainder.items[0]);
        remainder = quotient;
    }

    printableItems.push_back(remainder.items[0]);

    stringstream stream;
    stream << setfill('0');

    while (!printableItems.empty())
    {
        stream << printableItems.back();
        printableItems.pop_back();
        stream << setw(PRINTABLE_ITEM_WIDTH);
    }

    return stream.str();
}

void BigInt::readFrom(istream& stream)
{
    if (!isdigit(stream.peek()))
    {
        stream.setstate(ios::failbit);
        return;
    }

    *this = ZERO;

    do
    {
        *this *= 10;
        *this += (stream.get() - '0');
    }
    while (stream.good() && isdigit(stream.peek()));
}

bool BigInt::isEven() const
{
    return items[0] % 2 == 0;
}

/* returns the quotient and sets *this to the remainder */
BigInt BigInt::divMod(const BigInt& right)
{
    if(this == &right)
    {
        *this = BigInt::ZERO;
        return BigInt::ONE;
    }

    BigInt divisor(right);
    BigInt quotient;

    while (*this > divisor)
    {
        divisor <<= BITS_PER_ITEM;
    }

    while (divisor >= right)
    {
        quotient <<= 1;

        if (*this >= divisor)
        {
            *this -= divisor;
            ++quotient;
        }

        divisor >>= 1;
    }

    return quotient;
}

ssize_t BigInt::highestBitIndex() const
{
    uint32_t highOrderItem = items.back();
    ssize_t result = (items.size() - 1) * BITS_PER_ITEM - 1;

    while (highOrderItem > 0)
    {
        highOrderItem >>= 1;
        ++result;
    }

    return result;
}

bool BigInt::bit(size_t index) const
{
    size_t itemOffset = index % BITS_PER_ITEM;
    size_t blockOffset = index / BITS_PER_ITEM;

    if (blockOffset > items.size())
        return false;

    return (items[blockOffset] & (1 << itemOffset)) > 0;
}

void BigInt::normalize()
{
    while (!items.empty() && items.back() == 0)
    {
        items.pop_back();
    }

    if (items.empty())
    {
        items.push_back(0);
    }
}
