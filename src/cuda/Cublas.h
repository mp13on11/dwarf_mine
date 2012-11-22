#pragma once

#include <cublas_v2.h>

class Cublas {
public:
    Cublas();
    Cublas(Cublas&& other);
    Cublas(const Cublas&) = delete;
    Cublas& operator=(const Cublas& other) = delete;
    ~Cublas();

    int getVersion() const;
    void setStream(const cudaStream_t streamId);
    const cudaStream_t getStream() const;
    void setPointerMode(const cublasPointerMode_t mode);
    const cublasPointerMode_t getPointerMode() const;
    void setMatrix();
    void getMatrix();

private:
    cublasHandle_t handle;
};
