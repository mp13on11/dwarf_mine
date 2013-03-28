IF(BUILD_WITH_CUDA)
    ADD_SUBDIRECTORY(cuda-utils)
ENDIF()

ADD_SUBDIRECTORY(matrix)
ADD_SUBDIRECTORY(factorization_montecarlo)
ADD_SUBDIRECTORY(quadratic_sieve)
ADD_SUBDIRECTORY(othello_montecarlo)
