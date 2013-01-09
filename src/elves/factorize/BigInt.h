#pragma once

#include <inttypes.h>
#include <iosfwd>
#include <vector>
#include <string>

class BigInt;

std::ostream& operator<<(std::ostream& out, const BigInt& i);
std::istream& operator>>(std::istream& in, BigInt& i);

class BigInt
{
public:
    static const uint32_t MAX_ITEM = -1;

    static const BigInt& ZERO;
    static const BigInt& ONE;

    BigInt();
    BigInt(uint32_t value);
    explicit BigInt(const std::string& value);
    BigInt& operator=(uint32_t value);
    BigInt& operator=(const std::string& value);

    BigInt& operator++();
    BigInt operator++(int postfix);
    BigInt& operator--();
    BigInt operator--(int postfix);

    BigInt operator+(const BigInt& right) const;
    BigInt& operator+=(const BigInt& right);

    BigInt operator-(const BigInt& right) const;
    BigInt& operator-=(const BigInt& right);

    BigInt operator*(const BigInt& right) const;
    BigInt& operator*=(const BigInt& right);

    BigInt operator/(const BigInt& right) const;
    BigInt& operator/=(const BigInt& right);

    BigInt operator%(const BigInt& right) const;
    BigInt& operator%=(const BigInt& right);

    BigInt operator<<(uint32_t offset) const;
    BigInt& operator<<=(uint32_t offset);

    BigInt operator>>(uint32_t offset) const;
    BigInt& operator>>=(uint32_t offset);

    bool operator==(const BigInt& right) const;
    bool operator!=(const BigInt& right) const;
    bool operator>(const BigInt& right) const;
    bool operator>=(const BigInt& right) const;
    bool operator<(const BigInt& right) const;
    bool operator<=(const BigInt& right) const;

    bool isEven() const;
    bool isOdd() const;

    void readFrom(std::istream& stream);
    std::string toString() const;

private:
    static const uint32_t MAXIMUM_PRINTABLE_ITEM = 1000000000;
    static const uint32_t PRINTABLE_ITEM_WIDTH = 9;
    static const std::size_t BITS_PER_ITEM = 32;

    std::vector<uint32_t> items;

    void normalize();
    void reverse();
    void reverse(uint32_t &item);

    BigInt divMod(const BigInt& divisor);
};

inline std::ostream& operator<<(std::ostream& out, const BigInt& value)
{
    return out << value.toString();
}

inline std::istream& operator>>(std::istream& in, BigInt& value)
{
    value.readFrom(in);

    return in;
}

inline BigInt BigInt::operator+(const BigInt& right) const
{
    return BigInt(*this) += right;
}

inline BigInt BigInt::operator-(const BigInt& right) const
{
    return BigInt(*this) -= right;
}

inline BigInt BigInt::operator*(const BigInt& right) const
{
    return BigInt(*this) *= right;
}

inline BigInt BigInt::operator/(const BigInt& right) const
{
    return BigInt(*this) /= right;
}

inline BigInt BigInt::operator%(const BigInt& right) const
{
    return BigInt(*this) %= right;
}

inline BigInt BigInt::operator<<(uint32_t offset) const
{
    return BigInt(*this) <<= offset;
}

inline BigInt BigInt::operator>>(uint32_t offset) const
{
    return BigInt(*this) >>= offset;
}

inline BigInt& BigInt::operator++() // prefix
{
    return *this += ONE;
}

inline BigInt BigInt::operator++(int) // postfix
{
    BigInt result(*this);
    *this += ONE;

    return result;
}

inline BigInt& BigInt::operator--() // prefix
{
    return *this -= ONE;
}

inline BigInt BigInt::operator--(int) // postfix
{
    BigInt result(*this);
    *this -= ONE;

    return result;
}

inline bool BigInt::operator!=(const BigInt& right) const
{
    return !(*this == right);
}

inline bool BigInt::operator>(const BigInt& right) const
{
    return !(right >= *this);
}

inline bool BigInt::operator<(const BigInt& right) const
{
    return !(*this >= right);
}

inline bool BigInt::operator<=(const BigInt& right) const
{
    return right >= *this;
}

inline bool BigInt::isOdd() const
{
    return !isEven();
}
