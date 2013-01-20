MACRO(ADD_CXX_FLAGS FLAGS)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
ENDMACRO()

MACRO(ADD_COMMON_MSVC_PROPS EXECUTABLE_NAME)
    IF(MSVC)        
        SET_TARGET_PROPERTIES(${EXECUTABLE_NAME}
            PROPERTIES
            LINK_FLAGS "/NODEFAULTLIB:libcmt /SUBSYSTEM:CONSOLE")
    ENDIF()
ENDMACRO()

MACRO(ENABLE_CPP_11)
    IF(CMAKE_COMPILER_IS_GNUCXX)
        # Enable C++ 11 and threading for gcc
        ADD_CXX_FLAGS("-std=c++0x -pthread")
    ENDIF()
ENDMACRO()

