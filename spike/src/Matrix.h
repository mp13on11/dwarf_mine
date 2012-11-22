#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>

template<typename T>
class Matrix
{
public:
	Matrix(int rows, int columns);

	const T& operator()(int row, int column) const;
	T& operator()(int row, int column);

	const T* buffer() const;

private:
	int rows;
	int columns;
	std::vector<T> values;
};

template<typename T>
Matrix<T>::Matrix(int rows, int columns)
	: rows(rows), columns(columns), values(rows * columns)
{
}

template<typename T>
const T& Matrix<T>::operator()(int row, int column) const
{
	return values[row * columns + column];
}

template<typename T>
T& Matrix<T>::operator()(int row, int column)
{
	return values[row * columns + column];
}

template<typename T>
const T* Matrix<T>::buffer() const
{
	return values.data();
}

#endif /* MATRIX_H_ */
