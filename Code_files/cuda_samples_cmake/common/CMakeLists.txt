get_filename_component(target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(include_folder "${CMAKE_CURRENT_SOURCE_DIR}/inc")
set(src_folder "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB HEADERS_FILES "${include_folder}/*.h" "${include_folder}/*.hpp" "${include_folder}/*.hh")
file(GLOB SOURCES_FILES "${src_folder}/*.cpp" "${src_folder}/*.c" "${src_folder}/*.cu")


add_library(${target_name} STATIC
	${SOURCES_FILES}
	${HEADERS_FILES}
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
target_include_directories(${target_name} PUBLIC ${include_folder})

target_link_libraries(${target_name} PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/lib/x64/freeglut.lib
	${CMAKE_CURRENT_SOURCE_DIR}/lib/x64/glew64.lib
)
target_compile_features(${target_name} PUBLIC cxx_std_17)

