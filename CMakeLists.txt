cmake_minimum_required(VERSION 2.8)

project( 2D_ICP )
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-o3")

Find_Package(Eigen3 REQUIRED)
Find_Package(Sophus REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${EIGEN3_INCLUDE_DIR} ${Sophus_INCLUDE_DIRS}) 

add_executable(${PROJECT_NAME} 2d_icp.cpp)
target_link_libraries(${PROJECT_NAME} ${Sophus_LIBRARIES})