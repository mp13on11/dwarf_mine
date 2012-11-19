CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(13on11)

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)
SET(CMAKE_DEBUG_POSTFIX "d")

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

ADD_SUBDIRECTORY(src)
