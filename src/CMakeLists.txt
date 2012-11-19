SET(EXECUTABLE_NAME 13on11)

# Dependencies
#FIND_PACKAGE(OpenGL REQUIRED)
	
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}  
    ${CMAKE_CURRENT_BINARY_DIR} 
    ${PROJECT_SOURCE_DIR} 
    # Add dependencies' include dirs here
)

FILE(GLOB AUTO_SOURCES
	"*.h"
	"*.cpp"
)

# Create filters (in MSVC)
#SOURCE_GROUP("Group Name" FILES <FileNames>)
#SOURCE_GROUP(Source REGULAR_EXPRESSION "c$|cpp$|cu$")
#SOURCE_GROUP(Header REGULAR_EXPRESSION "hpp$|h$")

# Build executable out of sources
ADD_EXECUTABLE(${EXECUTABLE_NAME} 
    ${AUTO_SOURCES}
    # List additional source file lists here, if necessary
)

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME}
    # Add dependencies' libraries here
)    
    
IF(MSVC)  
	ADD_DEFINITIONS(/D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN)

    SET_TARGET_PROPERTIES(${EXECUTABLE_NAME}
        PROPERTIES
        LINK_FLAGS "/NODEFAULTLIB:libcmt /SUBSYSTEM:CONSOLE")

    INSTALL(TARGETS ${EXECUTABLE_NAME}
        DESTINATION . 
        CONFIGURATIONS Release)
ENDIF(MSVC)

