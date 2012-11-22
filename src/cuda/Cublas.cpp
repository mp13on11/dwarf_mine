#include "cublas_instance.h"

Cublas::Cublas() {
    cublasCreate(&handle);
}

Cublas::Cublas(Cublas&& other) {
    cublasDestroy(handle);
    handle = other.handle;
}

Cublas::~Cublas() {
    // TODO: Check if handle already destroyed
    cublasDestroy(handle);
}

int Cublas::getVersion() const {
    int result;
    cublasGetVersion(handle, &result);
    return result;
}

void Cublas::setStream(const cudaStream_t streamId) {

}

const cudaStream_t Cublas::getStream() const {

}

void Cublas::setPointerMode(const cublasPointerMode_t mode) {

}

const cublasPointerMode_t Cublas::getPointerMode() const {

}


