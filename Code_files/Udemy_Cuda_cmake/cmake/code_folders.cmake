cmake_minimum_required(VERSION 3.21)

get_filename_component(target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(include_folder "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(src_folder "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB CUDA_UTILS_INCLUDE_FILES "${UTILS_INCLUDE_FOLDER}/*.cu" "${UTILS_INCLUDE_FOLDER}/*.cuh" "${UTILS_INCLUDE_FOLDER}/*.h")
file(GLOB CUDA_UTILS_SRC_FILES "${UTILS_SRC_FOLDER}/*.cpp" "${UTILS_SRC_FOLDER}/*.c" "${UTILS_SRC_FOLDER}/*.cu")
file(GLOB INCLUDE_FILES "${include_folder}/*.h" "${include_folder}/*.hpp" "${include_folder}/*.hh" "${include_folder}/*.cuh")
file(GLOB SRC_FILES "${src_folder}/*.cpp" "${src_folder}/*.c" "${src_folder}/*.cu")


set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin")
# find_package(CUDA 10.2 REQUIRED)

message(STATUS "INCLUDE_FILES = ${INCLUDE_FILES}")
add_executable(${target_name} 
    ${SRC_FILES}
    ${CUDA_UTILS_SRC_FILES}
    ${INCLUDE_FILES} 
    ${CUDA_UTILS_INCLUDE_FILES}
    )
enable_language(CUDA)
set_target_properties(${target_name} PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "50"
)
if(VIPER_ENABLE_CUDA_PROFILER)
	target_compile_definitions(${target_name} PRIVATE ENABLE_CUDA_PROFILER=1)
endif()
target_include_directories(${target_name} PUBLIC 
    ${include_folder}
    ${OpenCV_INCLUDE_Folder} 
    ${UTILS_INCLUDE_FOLDER}
    )
target_compile_features(${target_name} PUBLIC cxx_std_17)

#add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OpenCV_DLL_Folder}/opencv_world430d.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
#add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OpenCV_DLL_Folder}/opencv_videoio_ffmpeg430_64.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
#target_link_libraries(${target_name} ${OpenCV_LIBS_Folder}/opencv_world430d.lib)


add_custom_command(TARGET ${target_name} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
COMMAND ${CMAKE_COMMAND} -E copy
    "$<IF:$<CONFIG:Debug>,${opencv_world430_dll_path_debug},${opencv_world430_dll_path_release}>"
    "${opencv_videoio_ffmpeg430_dll_path}"
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND_EXPAND_LISTS
)

target_link_libraries(${target_name} PUBLIC 
"$<IF:$<CONFIG:Debug>,${opencv_world430_lib_path_debug},${opencv_world430_lib_path_release}>"
)

