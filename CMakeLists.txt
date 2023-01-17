cmake_minimum_required(VERSION 3.10)

project(opticalFLow)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FLAGS  -Wall -Wextra -pedantic  -Werror)



find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ${PROJECT_NAME} main.cpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_compile_options(${PROJECT_NAME} PRIVATE ${FLAGS})