#pragma once

#include <main/MpiUtils.h>
#include <cstddef>

template<typename T>
class Matrix;

class MatrixSlice
{
public:
    MatrixSlice(
        NodeId responsibleNode,
        std::size_t startX,
        std::size_t startY,
        std::size_t columns,
        std::size_t rows
    );

    Matrix<float> extractSlice(const Matrix<float>& sourceMatrix, bool rowWise) const;
    void injectSlice(const Matrix<float>& sliceData, Matrix<float>& destMatrix) const;

    void send() const;
    void receive() const;

    NodeId getNodeId() const;
    std::size_t getColumns() const;
    std::size_t getRows() const;

private:
    NodeId nodeId;
    std::size_t x;
    std::size_t y;
    std::size_t columns;
    std::size_t rows;
};
