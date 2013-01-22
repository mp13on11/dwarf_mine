SET(EXECUTABLE_NAME test)

# Make mpiexec command name available to
# test code
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

FILE(GLOB CUDA_SOURCES
    "*.cu"
    "*.cuh"
)

IF(BUILD_WITH_CUDA)
    SET(CUDA_PROPAGATE_HOST_FLAGS 0)
    CUDA_COMPILE(CUDA_OBJECTS ${CUDA_SOURCES} ${LINKING_MODE} OPTIONS "--generate-code code=sm_21,arch=compute_20 --generate-code code=sm_20,arch=compute_20 --ptxas-options=-v")
ENDIF()

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} 
    ${AUTO_SOURCES}
    ${CUDA_OBJECTS}
)

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    "gtest"
    ${COMMON_LIBRARY}
    ${ELF_LIBRARIES}
    # required twice in order to resolve linker errors
    ${COMMON_LIBRARY}
    ${ELF_LIBRARIES}
    ${SUPPORT_LIBRARIES}
)    
    
ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
