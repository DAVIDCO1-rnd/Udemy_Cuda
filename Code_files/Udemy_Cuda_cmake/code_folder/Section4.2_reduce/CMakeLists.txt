get_filename_component(target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(includes_folder "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(sources_folder "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB HEADERS_FILES "${includes_folder}/*.h" "${includes_folder}/*.hpp" "${includes_folder}/*.hh")
file(GLOB SOURCES_FILES "${sources_folder}/*.cpp" "${sources_folder}/*.c" "${sources_folder}/*.cu")

set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin")
# find_package(CUDA 10.2 REQUIRED)


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
target_include_directories(${target_name} PUBLIC ${includes_folder})
target_compile_features(${target_name} PUBLIC cxx_std_17)

