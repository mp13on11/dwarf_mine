#include "Cublas.h"
#include "CublasError.h"
#include <string>

Cublas::Cublas()
{
    checkStatus(cublasCreate(&handle));
}

Cublas::Cublas(Cublas&& other)
{
    if (this != &other)
    {
        cublasDestroy(handle);
        handle = other.handle;
    }
}

Cublas::~Cublas()
{
    // TODO: Check if handle already destroyed
    cublasDestroy(handle);
}

void Cublas::Sgemm(
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const SMatrixPtr& A, int lda,
    const SMatrixPtr& B, int ldb,
    const float* beta,
    SMatrixPtr& C, int ldc)
{
    checkStatus(cublasSgemm(
        handle,
        transa,
        transb,
        m, n, k,
        alpha,
        A.get(), lda,
        B.get(), ldb,
        beta,
        C.get(), ldc
    ));
}

void Cublas::checkStatus(cublasStatus_t status) const
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw CublasError(status);
    }
}

