#pragma once

#include <cublas_v2.h>
#include "Memory.h"
#include <string>

class Cublas
{
public:
    typedef CudaUtils::Memory<float> SMatrixPtr;

    Cublas();
    Cublas(Cublas&& other);
    Cublas(const Cublas&) = delete;
    Cublas& operator=(const Cublas& other) = delete;

    ~Cublas();

    //
    // CUBLAS helper functions
    template<typename MatrixType>
    void setMatrix(int rows, int cols, int elemSize, const MatrixType* A, int lda, CudaUtils::Memory<MatrixType>& B, int ldb)
    {
        checkStatus(cublasSetMatrix(
            rows,
            cols,
            elemSize,
            A,
            lda,
            B.get(),
            ldb
        ));
    }

    template<typename MatrixType>
    void getMatrix(int rows, int cols, int elemSize, const CudaUtils::Memory<MatrixType>& A, int lda, MatrixType* B, int ldb)
    {
        checkStatus(cublasGetMatrix(
            rows,
            cols,
            elemSize,
            A.get(),
            lda,
            B,
            ldb
        ));
    }

    //
    // CUBLAS
    void Sgemm(
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const float* alpha,
        const SMatrixPtr& A, int lda,
        const SMatrixPtr& B, int ldb,
        const float* beta,
        SMatrixPtr& C, int ldc
    );

private:
    void checkStatus(cublasStatus_t status) const;

    cublasHandle_t handle;
};
