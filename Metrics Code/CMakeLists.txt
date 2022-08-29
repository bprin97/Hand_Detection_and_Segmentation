cmake_minimum_required(VERSION 3.0.0)
project(ProjectMetrics VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV  REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/src )
add_library(Metrics src/Metrics.cpp)
target_include_directories(Metrics PUBLIC ${PROJECT_SOURCE_DIR})

add_executable(ProjectMetrics src/main.cpp)

target_link_libraries( ${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} Metrics)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)