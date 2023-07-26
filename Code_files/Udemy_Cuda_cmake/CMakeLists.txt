cmake_minimum_required(VERSION 3.21)
project(my_solution VERSION 1.0.0)

option(VIPER_CONFIG_VERBOSE "Use verbose detail when configuring" ON)
option(VIPER_ENABLE_CUDA_PROFILER "Enable NVIDIA nSight profiler support" ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


if (CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(VIPER_ARCH "x64")
else (CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(VIPER_ARCH "x86")
endif()

set(CUDA_BINARY_DIR_OUTPUT ${CMAKE_BINARY_DIR}/bin/${VIPER_ARCH}/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CUDA_BINARY_DIR_OUTPUT})


message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "CUDA_BINARY_DIR_OUTPUT = ${CUDA_BINARY_DIR_OUTPUT}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")

set(SRC_FOLDERS ${CMAKE_SOURCE_DIR}/code_folder)

SUBDIRLIST(SUBDIRS ${SRC_FOLDERS})

FOREACH(subdir ${SUBDIRS})
  add_subdirectory(${SRC_FOLDERS}/${subdir})
ENDFOREACH()
