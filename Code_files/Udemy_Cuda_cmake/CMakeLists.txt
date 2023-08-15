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

set(OpenCV_main_folder "${CMAKE_CURRENT_SOURCE_DIR}/opencv_4.3.0")
set(OpenCV_INCLUDE_Folder "${OpenCV_main_folder}/build/include")
set(OpenCV_LIBS_Folder "${OpenCV_main_folder}/build/x64/vc15/lib")
set(OpenCV_DLL_Folder "${OpenCV_main_folder}/build/x64/vc15/bin")

set(opencv_world430_lib_path_debug "${OpenCV_LIBS_Folder}/opencv_world430d.lib")
set(opencv_world430_lib_path_release "${OpenCV_LIBS_Folder}/opencv_world430.lib")

set(opencv_world430_dll_path_debug "${OpenCV_DLL_Folder}/opencv_world430d.dll")
set(opencv_world430_dll_path_release "${OpenCV_DLL_Folder}/opencv_world430.dll")

set(opencv_videoio_ffmpeg430_dll_path "${OpenCV_DLL_Folder}/opencv_videoio_ffmpeg430_64.dll")




message(STATUS "OpenCV_main_folder = ${OpenCV_main_folder}")

message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS "CUDA_BINARY_DIR_OUTPUT = ${CUDA_BINARY_DIR_OUTPUT}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")

set(SRC_FOLDERS ${CMAKE_SOURCE_DIR}/code_folder)
set(CUDA_UTILS_SRC_FOLDER ${CMAKE_SOURCE_DIR}/cuda_utils/src)
set(CUDA_UTILS_INCLUDE_FOLDER ${CMAKE_SOURCE_DIR}/cuda_utils/include)

SUBDIRLIST(SUBDIRS ${SRC_FOLDERS})

FOREACH(subdir ${SUBDIRS})
  add_subdirectory(${SRC_FOLDERS}/${subdir})
ENDFOREACH()
