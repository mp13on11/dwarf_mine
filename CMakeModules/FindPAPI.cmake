#
# Find PAPI libraries
#

FIND_PACKAGE(CUDA)
IF (CUDA_FOUND)	

	SET(PAPI_POSSIBLE_INCLUDE_PATHS
	  /usr/include
	  /usr/local/include
	)

	SET(PAPI_POSSIBLE_LIBRARY_PATHS
	  /usr/lib
	  /usr/local/lib
	)

	SET(CUPTI_LIBRARY_PATHS
	  ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
        )

	FIND_PATH(PAPI_INCLUDE_DIR NAMES papi.h PATHS ${PAPI_POSSIBLE_INCLUDE_PATHS})

	FIND_LIBRARY(PAPI_LIBRARY
          NAMES libpapi.so libcupti.so 
          PATHS ${PAPI_POSSIBLE_LIBRARY_PATHS})

	FIND_LIBRARY(CUPTI_LIBRARY
          NAMES libcupti.so 
          PATHS ${CUPTI_LIBRARY_PATHS})

#	INCLUDE(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
	FIND_PACKAGE_HANDLE_STANDARD_ARGS(PAPI DEFAULT_MSG PAPI_LIBRARY PAPI_INCLUDE_DIR)

	SET(PAPI_LIBRARIES ${PAPI_LIBRARY} ${CUPTI_LIBRARY})
	MESSAGE(STATUS "FUUUU" ${PAPI_LIBRARIES})

	MARK_AS_ADVANCED(
	  PAPI_INCLUDE_DIR
	  PAPI_LIBRARIES
	)

ENDIF (CUDA_FOUND)
