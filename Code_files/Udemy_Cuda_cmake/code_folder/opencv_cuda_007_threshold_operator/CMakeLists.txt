get_filename_component(target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(includes_folder "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(sources_folder "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB HEADERS_FILES "${includes_folder}/*.h" "${includes_folder}/*.hpp" "${includes_folder}/*.hh" "${includes_folder}/*.cuh" "${CUDA_UTILS_FOLDERS}/*.cu" "${CUDA_UTILS_FOLDERS}/*.cuh")
file(GLOB SOURCES_FILES "${sources_folder}/*.cpp" "${sources_folder}/*.c" "${sources_folder}/*.cu")

set(OpenCV_main_folder "../../../../opencv_4.3.0")
set(OpenCV_INCLUDE_Folder "${CMAKE_CURRENT_SOURCE_DIR}/${OpenCV_main_folder}/build/include")
set(OpenCV_LIBS_Folder "${CMAKE_CURRENT_SOURCE_DIR}/${OpenCV_main_folder}/build/x64/vc15/lib")
set(OpenCV_DLL_Folder "${CMAKE_CURRENT_SOURCE_DIR}/${OpenCV_main_folder}/build/x64/vc15/bin")



set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin")
find_package(CUDA 10.2 REQUIRED)


add_executable(${target_name} ${SOURCES_FILES} ${CUDA_SOURCES_FILES} ${HEADERS_FILES})
enable_language(CUDA)
set_target_properties(${target_name} PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "30"
)
if(VIPER_ENABLE_CUDA_PROFILER)
	target_compile_definitions(${target_name} PRIVATE ENABLE_CUDA_PROFILER=1)
endif()
target_include_directories(${target_name} PUBLIC ${includes_folder} ${OpenCV_INCLUDE_Folder} ${CUDA_UTILS_FOLDERS})
target_compile_features(${target_name} PUBLIC cxx_std_17)

add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OpenCV_DLL_Folder}/opencv_world430d.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_custom_command(TARGET ${target_name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OpenCV_DLL_Folder}/opencv_videoio_ffmpeg430_64.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
target_link_libraries(${target_name} ${OpenCV_LIBS_Folder}/opencv_world430d.lib)

