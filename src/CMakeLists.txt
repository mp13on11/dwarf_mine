# Add global dependencies here
FIND_PACKAGE(Boost REQUIRED)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/elves/
    # Add dependencies' include dirs here
)

ADD_SUBDIRECTORY(main)
ADD_SUBDIRECTORY(elves)
