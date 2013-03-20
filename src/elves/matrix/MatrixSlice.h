#pragma once

#include <cstddef>

template<typename T>
class Matrix;

class MatrixSlice
{
public:
    MatrixSlice() = default;
    MatrixSlice(
        std::size_t startX,
        std::size_t startY,
        std::size_t columns,
        std::size_t rows
    );
    MatrixSlice(
        int responsibleNode,
        std::size_t startX,
        std::size_t startY,
        std::size_t columns,
        std::size_t rows
    );

    Matrix<float> extractSlice(const Matrix<float>& sourceMatrix, bool rowWise) const;
    void injectSlice(const Matrix<float>& sliceData, Matrix<float>& destMatrix) const;

    int getNodeId() const;
    void setNodeId(const int nodeId);
    std::size_t getStartX() const;
    std::size_t getStartY() const;
    std::size_t getColumns() const;
    std::size_t getRows() const;

private:
    int nodeId;
    std::size_t x;
    std::size_t y;
    std::size_t columns;
    std::size_t rows;
};
