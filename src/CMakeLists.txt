# Add global dependencies here
FIND_PACKAGE(Boost REQUIRED)

INCLUDE(Utils)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/lib/
    # Add dependencies' include dirs here
)

ADD_SUBDIRECTORY(lib/benchmark)
ADD_SUBDIRECTORY(lib/tools)
ADD_SUBDIRECTORY(gold)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(mpi)
#ADD_SUBDIRECTORY(smp)
