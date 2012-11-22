#pragma once

#include <vector>

using namespace std;

template<typename T>
class Matrix
{
public:
    Matrix(size_t rows=0, size_t columns=0);

    size_t rows() const;
    size_t columns() const;

    const T& operator()(size_t row, size_t column) const;
    T& operator()(size_t row, size_t column);

    const T* buffer() const;

private:
    size_t _rows;
    size_t _columns;
    std::vector<T> values;
};

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t columns)
    : _rows(rows), _columns(columns), values(rows * columns)
{
}

template<typename T>
const T& Matrix<T>::operator()(size_t row, size_t column) const
{
    return values[row * _columns + column];
}

template<typename T>
T& Matrix<T>::operator()(size_t row, size_t column)
{
    return values[row * _columns + column];
}

template<typename T>
const T* Matrix<T>::buffer() const
{
    return values.data();
}

template<typename T>
size_t Matrix<T>::rows() const
{
    return _rows;
}

template<typename T>
size_t Matrix<T>::columns() const
{
    return _columns;
}
