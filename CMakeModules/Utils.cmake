MACRO(ADD_CXX_FLAGS FLAGS)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
ENDMACRO()

MACRO(REGISTER_ELF_LIB LIB_NAME)
    IF ("${ELF_LIBRARIES}" STREQUAL "")
        SET(NEW_VALUE ${LIB_NAME})
    ELSE()
        SET(NEW_VALUE "${ELF_LIBRARIES};${LIB_NAME}")
    ENDIF()

    SET(ELF_LIBRARIES ${NEW_VALUE} CACHE STRING INTERNAL FORCE)
ENDMACRO()

MACRO(ADD_COMMON_MSVC_PROPS EXECUTABLE_NAME)
    IF(MSVC)        
        SET_TARGET_PROPERTIES(${EXECUTABLE_NAME}
            PROPERTIES
            LINK_FLAGS "/NODEFAULTLIB:libcmt /SUBSYSTEM:CONSOLE")
    ENDIF()
ENDMACRO()
