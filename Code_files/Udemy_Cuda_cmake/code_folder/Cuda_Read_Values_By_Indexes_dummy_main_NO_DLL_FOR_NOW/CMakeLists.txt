get_filename_component(target_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(main_includes_folder "${CMAKE_SOURCE_DIR}/include")
set(sources_folder "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB HEADERS_FILES "${includes_folder}/*.h")
file(GLOB SOURCES_FILES "${sources_folder}/*.cpp")
add_executable(${target_name} ${SOURCES_FILES} ${HEADERS_FILES})
target_include_directories(${target_name} PUBLIC ${main_includes_folder})
#target_link_libraries(${target_name} PUBLIC 

#)




