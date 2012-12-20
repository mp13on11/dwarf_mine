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
    
    void addPadding(size_t multiple);
    void resizeLossy(std::size_t newRows, std::size_t newColumns);

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

#include <iostream>

template<typename T>
void Matrix<T>::resizeLossy(std::size_t newRows, std::size_t newColumns)
{
    if (newRows == _rows && newColumns == _columns) 
    {
        return;
    }

    std::vector<float> newValues(newColumns*newRows);
    
    for (std::size_t i = 0; i < newRows; ++i)
    {
        for (std::size_t j = 0; j < newColumns; ++j)
        {
            newValues.at(i*newColumns+j) = values.at(i*_columns+j);
        }  
    }    

    values = newValues;
    _columns = newColumns;
    _rows = newRows;
}

template<typename T>
void Matrix<T>::addPadding(std::size_t multiple)
{
    //std::cout << _rows << ", " << _columns << std::endl;    

    std::size_t columnRest = _columns % multiple;
    std::size_t rowRest = _rows % multiple;

    if (columnRest == 0 && rowRest == 0)
        return;

    std::size_t newColumns = (columnRest == 0)? _columns : (_columns/multiple + 1) * multiple;
    std::size_t newRows = (rowRest == 0)? _rows : (_rows/multiple + 1) * multiple;

    //std::cout << newRows << ", " << newColumns << std::endl;

    std::vector<float> newValues(newColumns*newRows, 0);
    
    for (std::size_t i = 0; i < _rows; ++i)
    {
        for (std::size_t j = 0; j < _columns; ++j)
        {
            newValues.at(i*newColumns+j) = values.at(i*_columns+j);
        }  
    }

    values = newValues;

    _columns = newColumns;
    _rows = newRows;
}

