CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

PROJECT(DOO-SABIN_REGRESSION)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

SET(EIGEN_INCLUDE_DIR "" CACHE PATH "")
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

SET(COMMON_CPP_INCLUDE_DIR "" CACHE PATH "")
INCLUDE_DIRECTORIES(${COMMON_CPP_INCLUDE_DIR})

SET(DOOSABIN_INCLUDE_DIR "" CACHE PATH "")
INCLUDE_DIRECTORIES(${DOOSABIN_INCLUDE_DIR})

# Ceres
SET(ACML_INCLUDE "" CACHE PATH "")
SET(ACML_LIB "" CACHE FILEPATH "")
SET(BLAS_LAPACK_LIBRARIES ${ACML_LIB})

SET(CERES_INCLUDE_DIR "" CACHE PATH "")
SET(CERES_INTERNAL_DIR "" CACHE PATH "")
SET(CERES_LIB "" CACHE FILEPATH "")
SET(CERES_LIB_DEBUG "" CACHE FILEPATH "")

SET(GFLAGS_INCLUDE_DIR "" CACHE PATH "")
SET(LIBGFLAGS_LIB "" CACHE FILEPATH "")
SET(GLOG_INCLUDE_DIR "" CACHE PATH "")
SET(LIBGLOG_LIB "" CACHE FILEPATH "")

SET(SUITESPARSE_INCLUDE_DIR "" CACHE PATH "")
SET(SUITESPARSEQR_INCLUDE_DIR "" CACHE PATH "")
SET(CHOLMOD_INCLUDE_DIR "" CACHE FILEPATH "")
SET(CHOLMOD_LIB "" CACHE FILEPATH "")

SET(CERES_INCLUDE_DIRS ${CERES_INCLUDE_DIR}
                       ${CERES_INTERNAL_DIR}
                       ${GFLAGS_INCLUDE_DIR}
                       ${GLOG_INCLUDE_DIR}
                       ${SUITESPARSE_INCLUDE_DIR}
                       ${SUITESPARSEQR_INCLUDE_DIR}
                       ${CHOLMOD_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})

SET(CERES_LIBRARIES optimized
                    ${CERES_LIB}
                    debug
                    ${CERES_LIB_DEBUG})
LIST(APPEND CERES_LIBRARIES ${BLAS_LAPACK_LIBRARIES}
                            ${CHOLMOD_LIB}
                            ${LIBGFLAGS_LIB}
                            ${LIBGLOG_LIB})

FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

# Required by Protobuf.
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

# Define PROTOBUF_GENERATE_PYTHON to generate Python Protobuf files
# in the source tree.
# This is done so that execution of visualiation Python scripts is
# straightforward.
FUNCTION(PROTOBUF_GENERATE_PYTHON MODULES)
  IF(NOT ARGN)
    MESSAGE(SEND_ERROR "Error: PROTOBUF_GENERATE_PYTHON() called without any proto files")
  ENDIF()

  SET(${MODULES})

  FOREACH(FIL ${ARGN})
    GET_FILENAME_COMPONENT(ABS_FIL ${FIL} ABSOLUTE)
    GET_FILENAME_COMPONENT(FIL_WE ${FIL} NAME_WE)
    LIST(APPEND ${MODULES} "${CMAKE_SOURCE_DIR}/${FIL_WE}_pb2.py")

    ADD_CUSTOM_COMMAND(
      OUTPUT "${CMAKE_SOURCE_DIR}/${FIL_WE}_pb2.py"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --python_out ${CMAKE_SOURCE_DIR}/../python --proto_path ${CMAKE_SOURCE_DIR} ${ABS_FIL}
      DEPENDS ${ABS_FIL}
      COMMENT "Running Python protocol buffer compiler on ${FIL}"
      VERBATIM)
  ENDFOREACH()

  SET(${MODULES} ${${MODULES}} PARENT_SCOPE)
ENDFUNCTION()

PROTOBUF_GENERATE_CPP(DOOSABIN_REGRESSION_PROTO_SRCS
                      DOOSABIN_REGRESSION_PROTO_HDRS
                      doosabin_regression.proto)
PROTOBUF_GENERATE_PYTHON(DOOSABIN_REGRESSION_PROTO_PYTHON
                         doosabin_regression.proto)

SET(DOOSABIN_REGRESSION_SRCS
    ceres_surface.cpp
    ceres_surface.h
    doosabin_regression.cpp
    patch_index_encoding.cpp
    patch_index_encoding.h
    surface.h
    ${DOOSABIN_REGRESSION_PROTO_SRCS}
    ${DOOSABIN_REGRESSION_PROTO_HDRS}
    ${DOOSABIN_REGRESSION_PROTO_PYTHON})
ADD_EXECUTABLE(doosabin_regression ${DOOSABIN_REGRESSION_SRCS})
TARGET_LINK_LIBRARIES(doosabin_regression
                      ${CERES_LIBRARIES}
                      ${PROTOBUF_LIBRARIES})