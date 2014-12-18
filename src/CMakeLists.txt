CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
PROJECT(DOO-SABIN_REGRESSION)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

MACRO(EXPECT_FILES VARIABLE)
  FOREACH(EXPECTED_FILE ${ARGN})
    IF (NOT EXISTS ${${VARIABLE}}/${EXPECTED_FILE})
      MESSAGE(FATAL_ERROR
        "Caller defined ${VARIABLE}: ${${VARIABLE}} does not contain "
        "${EXPECTED_FILE}.")
    ENDIF (NOT EXISTS ${${VARIABLE}}/${EXPECTED_FILE})
  ENDFOREACH()
ENDMACRO(EXPECT_FILES)

SET(RAPID_JSON_INCLUDE_DIR "" CACHE PATH "")
EXPECT_FILES(RAPID_JSON_INCLUDE_DIR rapidjson/document.h
                                    rapidjson/filestream.h
                                    rapidjson/prettywriter.h)
INCLUDE_DIRECTORIES(${RAPID_JSON_INCLUDE_DIR})

SET(COMMON_CPP_INCLUDE_DIR "" CACHE PATH "")
EXPECT_FILES(COMMON_CPP_INCLUDE_DIR Ceres/compose_cost_functions.h
                                    Math/linalg.h)
INCLUDE_DIRECTORIES(${COMMON_CPP_INCLUDE_DIR})

SET(DOOSABIN_INCLUDE_DIR "" CACHE PATH "")
EXPECT_FILES(DOOSABIN_INCLUDE_DIR doosabin.h)
INCLUDE_DIRECTORIES(${DOOSABIN_INCLUDE_DIR})

# Ceres
FIND_PACKAGE(Ceres REQUIRED)

SET(GFLAGS_INCLUDE_DIR "" CACHE PATH "")
EXPECT_FILES(GFLAGS_INCLUDE_DIR gflags/gflags.h)

SET(GFLAGS_LIBRARY "" CACHE FILEPATH "")
# From FindGlog.cmake:
STRING(TOLOWER "${GFLAGS_LIBRARY}" LOWERCASE_GFLAGS_LIBRARY)
IF (NOT "${LOWERCASE_GFLAGS_LIBRARY}" MATCHES ".*gflags[^/]*")
    MESSAGE(FATAL_ERROR
      "Caller defined GFLAGS_LIBRARY: ${GFLAGS_LIBRARY} does not match "
      "gflags.")
ENDIF (NOT "${LOWERCASE_GFLAGS_LIBRARY}" MATCHES ".*gflags[^/]*")

INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS}
                    ${GFLAGS_INCLUDE_DIR})

SET(DOOSABIN_REGRESSION_SRCS
    ceres_surface.cpp
    ceres_surface.h
    doosabin_regression.cpp
    patch_index_encoding.cpp
    patch_index_encoding.h
    surface.h)
ADD_EXECUTABLE(doosabin_regression ${DOOSABIN_REGRESSION_SRCS})
TARGET_LINK_LIBRARIES(doosabin_regression
                      ${CERES_LIBRARIES}
                      ${GFLAGS_LIBRARY})
