CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(dwarf_mine)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "limited configs" FORCE)
SET(CMAKE_DEBUG_POSTFIX "d")

INCLUDE(Utils)

# Enable Vampir profiling.
#
# WARNING: Do not set this option manually - use the script provided
# in /utilities/use_vampir_trace.sh
OPTION(USE_VAMPIR_TRACE OFF)

# Enable C++ 11 and threading for gcc
IF(CMAKE_COMPILER_IS_GNUCXX)
    ADD_CXX_FLAGS("-std=c++0x -pthread")
ENDIF()

IF(MSVC)
    MESSAGE(WARNING
        "This project has never been tested on Windows. Most of the code should 
        be portable, but there might be compiler errors or other problems. 
        Don't expect it to simply work."
    )
ENDIF()

# Enable strict warnings
IF(MSVC)
    ADD_CXX_FLAGS("/W4")
ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    ADD_CXX_FLAGS("-Wall -Wextra -W -pedantic")

    # Warnings as errors, unless disabled by the user
    OPTION(WARNINGS_AS_ERRORS "Set to off to disable treating warnings as errors" ON)
    IF(WARNINGS_AS_ERRORS)
        ADD_CXX_FLAGS("-Werror")
    ENDIF()
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
        ADD_CXX_FLAGS("/MP")
    ENDIF()
ENDIF()

# Add sources from subdirectories
ADD_SUBDIRECTORY(gtest-1.6.0)
ADD_SUBDIRECTORY(src)

IF(USE_VAMPIR_TRACE)
    ADD_CUSTOM_TARGET(symbols ALL
        COMMAND find . -name *.so > /tmp/dwarf_mine.symbols.init\; tr \"\\n\" \" \" < /tmp/dwarf_mine.symbols.init | xargs nm ./src/main/dwarf_mine > /tmp/dwarf_mine.symbols
    )
    ADD_DEPENDENCIES(symbols dwarf_mine)
ENDIF()
