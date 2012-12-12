CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(dwarf_mine)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)
SET(CMAKE_DEBUG_POSTFIX "d")

INCLUDE(Utils)

# Enable strict warnings
IF(MSVC)
    SET(CMAKE_CXX_FLAGS "/W4 ${CMAKE_CXX_FLAGS}")
ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "-Wall -Wextra -W -pedantic ${CMAKE_CXX_FLAGS}")
ENDIF()

# Require Visual Studio 2012 or newer,
# older VS versions do not support C++ 11
IF(MSVC AND NOT MSVC11)
    MESSAGE(FATAL_ERROR "This project needs at least Visual Studio 2012")
ENDIF()

# Visual Studio specific settings
IF(MSVC)
    ADD_DEFINITIONS(/D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN)

    # Provide an option to enable multithreaded compilation in Visual Studio
    OPTION(WIN32_USE_MP "Set to ON to build with the /MP option." ON)
    IF(WIN32_USE_MP)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    ENDIF()
ENDIF()

ADD_SUBDIRECTORY(gtest-1.6.0)
ADD_SUBDIRECTORY(src)
