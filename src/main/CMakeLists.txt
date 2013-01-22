SET(EXECUTABLE_NAME dwarf_mine)

# Dependencies
INCLUDE_DIRECTORIES(
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
    "*.h"
    "*.cpp"
)

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} ${AUTO_SOURCES})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
    ${COMMON_LIBRARY}
    ${ELF_LIBRARIES}
    ${SUPPORT_LIBRARIES}
)

ADD_COMMON_MSVC_PROPS(${EXECUTABLE_NAME})
