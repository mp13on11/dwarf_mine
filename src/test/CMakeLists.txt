SET(EXECUTABLE_NAME test)

ENABLE_CPP_11()

# Dependencies
    
ADD_DEFINITIONS("-DMPIEXEC=\"${MPIEXEC}\"")

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/gtest-1.6.0/include/
    # Add dependencies' include dirs here
    ${GMP_INCLUDE_DIR}
    ${CUDA_TOOLKIT_INCLUDE}
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# We need all files from the main executable as well, but not Main.cpp,
# since we have our own main()-function.

SET(MAIN_PREFIX "../main")

FILE(GLOB MAIN_SOURCES
    "${MAIN_PREFIX}/*.h"
    "${MAIN_PREFIX}/*.cpp"
)
FILE(GLOB TO_OMIT "${MAIN_PREFIX}/Main.cpp")
LIST(REMOVE_ITEM MAIN_SOURCES ${TO_OMIT})

FILE(GLOB CUDA_SOURCES
    "*.cu"
    "*.cuh"
)

SET(CUDA_PROPAGATE_HOST_FLAGS 0)
CUDA_COMPILE(CUDA_OBJECTS ${CUDA_SOURCES} ${LINKING_MODE} OPTIONS "--generate-code code=sm_21,arch=compute_20 --generate-code code=sm_20,arch=compute_20 --ptxas-options=-v")

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} 
    ${AUTO_SOURCES} 
    ${MAIN_SOURCES}
    ${CUDA_OBJECTS}
)

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${ELF_LIBRARIES}
    ${MPI_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GMP_LIBRARIES}
    ${GMPXX_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    "gtest"
)    
    
ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
