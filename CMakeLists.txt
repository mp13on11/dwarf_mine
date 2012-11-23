CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(spike)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)
SET(CMAKE_DEBUG_POSTFIX "d")

IF(MSVC)
    SET(CMAKE_CXX_FLAGS "/W4 ${CMAKE_CXX_FLAGS}")
ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "-Wall -Wextra -W -pedantic ${CMAKE_CXX_FLAGS}")
ENDIF()

IF(MSVC AND NOT MSVC10)
    # Older VS versions do not support C++ 11
    MESSAGE(FATAL_ERROR "This project needs at least Visual Studio 2010")
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
    # Enable C++ 11 and threading for gcc
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -pthread")

    # Provide syntax highlighting for C++ 11 in Eclipse
    ADD_DEFINITIONS("-D__GXX_EXPERIMENTAL_CXX0X__")
ENDIF()

# Provide an option to enable multithreaded compilation in Visual Studio
IF(MSVC)
    OPTION(WIN32_USE_MP "Set to ON to build with the /MP option (Visual Studio 2005 and above)." ON)
    IF(WIN32_USE_MP)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    ENDIF()
ENDIF()

ADD_SUBDIRECTORY(gtest-1.6.0)
ADD_SUBDIRECTORY(src)
