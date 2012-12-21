#pragma once

#include <vector>
#include <memory>

template<typename T>
class Matrix
{
public:

    explicit Matrix(std::size_t rows=0, std::size_t columns=0);
    Matrix(std::size_t rows, std::size_t columns, std::vector<T>&& data);

    std::size_t rows() const;
    std::size_t columns() const;

    const T& operator()(std::size_t row, std::size_t column) const;
    T& operator()(std::size_t row, std::size_t column);

    const T* buffer() const;
    T* buffer();

private:
    std::size_t _rows;
    std::size_t _columns;
    std::vector<T> values;
};

template<typename T>
Matrix<T>::Matrix(std::size_t rows, std::size_t columns, std::vector<T>&& data)
    : _rows(rows), _columns(columns), values(std::move(data))
{
}

template<typename T>
Matrix<T>::Matrix(std::size_t rows, std::size_t columns)
    : _rows(rows), _columns(columns), values(rows * columns)
{
}

template<typename T>
const T& Matrix<T>::operator()(std::size_t row, std::size_t column) const
{
    return values[row * _columns + column];
}

template<typename T>
T& Matrix<T>::operator()(std::size_t row, std::size_t column)
{
    return values[row * _columns + column];
}

template<typename T>
const T* Matrix<T>::buffer() const
{
    return values.data();
}

template<typename T>
T* Matrix<T>::buffer()
{
    return values.data();
}

template<typename T>
std::size_t Matrix<T>::rows() const
{
    return _rows;
}

template<typename T>
std::size_t Matrix<T>::columns() const
{
    return _columns;
}
