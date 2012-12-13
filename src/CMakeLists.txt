# Add global dependencies here
FIND_PACKAGE(Boost REQUIRED)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/
    ${PROJECT_SOURCE_DIR}/src/elves/
    # Add dependencies' include dirs here
)

ADD_SUBDIRECTORY(elves)
ADD_SUBDIRECTORY(main)
