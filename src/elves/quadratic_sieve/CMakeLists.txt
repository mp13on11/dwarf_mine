SET(LIBRARY_NAME quadratic-sieve)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

REGISTER_ELF_LIB(${LIBRARY_NAME})

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

IF(CMAKE_COMPILER_IS_GNUCXX OR USE_VAMPIR_TRACE)
    ADD_CXX_FLAGS("-O3 -fopenmp -funroll-loops")
ENDIF()

ADD_LIBRARY(${LIBRARY_NAME} STATIC ${AUTO_SOURCES})
ADD_COMMON_MSVC_PROPS(${LIBRARY_NAME})

ADD_SUBDIRECTORY(smp)

IF(BUILD_WITH_CUDA)
    ADD_SUBDIRECTORY(cuda)
ENDIF()
