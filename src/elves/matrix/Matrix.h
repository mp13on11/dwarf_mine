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

#include <vector>
#include <memory>
#include <cassert>

template<typename T>
class Matrix
{
public:
    explicit Matrix(std::size_t rows=0, std::size_t columns=0);
    Matrix(std::size_t rows, std::size_t columns, std::vector<T>&& data);

    bool empty() const;

    std::size_t rows() const;
    std::size_t columns() const;

    const T& operator()(std::size_t row, std::size_t column) const;
    T& operator()(std::size_t row, std::size_t column);

    const T* buffer() const;
    T* buffer();

    Matrix<T> transposed() const;

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
Matrix<T> Matrix<T>::transposed() const
{
    Matrix<T> result(_columns, _rows);

    for (std::size_t i=0; i<_rows; i++)
    {
        for (std::size_t j=0; j<_columns; j++)
        {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

template<typename T>
const T& Matrix<T>::operator()(std::size_t row, std::size_t column) const
{
    return values[row * _columns + column];
}

template<typename T>
T& Matrix<T>::operator()(std::size_t row, std::size_t column)
{
    assert(row < rows());
    assert(column < columns());
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

template<typename T>
bool Matrix<T>::empty() const
{
    return _rows == 0 || _columns == 0;
