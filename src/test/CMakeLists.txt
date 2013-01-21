SET(EXECUTABLE_NAME test)

# Make mpiexec command name available to
# test code
ADD_DEFINITIONS("-DMPIEXEC=\"${MPIEXEC}\"")

# Dependencies
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/gtest-1.6.0/include/
    # Add dependencies' include dirs here
    ${GMP_INCLUDE_DIR}
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

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} 
    ${AUTO_SOURCES} 
    ${MAIN_SOURCES}
)

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${ELF_LIBRARIES}
    ${MPI_LIBRARIES}
    ${Boost_LIBRARIES}
    ${GMP_LIBRARIES}
    ${GMPXX_LIBRARIES}
    "gtest"
)    
    
ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})