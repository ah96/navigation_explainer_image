cmake_minimum_required(VERSION 3.0.2)
project(navigation_explainer)

set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  std_srvs
  base_local_planner
  costmap_2d
  costmap_converter
  cmake_modules
  dynamic_reconfigure
  geometry_msgs
  interactive_markers
  message_generation
  nav_core
  nav_msgs
  mbf_costmap_core
  mbf_msgs
  roscpp
  std_msgs
  pluginlib
  tf2
  tf2_eigen
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

message(STATUS "System: ${CMAKE_SYSTEM}")
## System dependencies are found with CMake's conventions
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)
message(STATUS "${CMAKE_MODULE_PATH}")
find_package(Boost REQUIRED COMPONENTS system thread graph)
find_package(SUITESPARSE REQUIRED)
find_package(G2O REQUIRED)

# Eigen3 FindScript Backward compatibility (ubuntu saucy)
# Since FindEigen.cmake is deprecated starting from jade.
if (EXISTS "FindEigen3.cmake")
  find_package(Eigen3 REQUIRED)
  set(Eigen_INCLUDE_DIRS ${Eigen3_INCLUDE_DIRS})
elseif (EXISTS "FindEigen.cmake")
  find_package(Eigen REQUIRED)
elseif (EXISTS "FindEigen.cmake")
  message(WARNING "No findEigen cmake script found. You must provde one of them,
  e.g. by adding it to ${PROJECT_SOURCE_DIR}/cmake_modules.")
endif (EXISTS "FindEigen3.cmake")

set(EXTERNAL_INCLUDE_DIRS ${Eigen_INCLUDE_DIRS} ${SUITESPARSE_INCLUDE_DIRS} ${G2O_INCLUDE_DIR})
set(EXTERNAL_LIBS ${SUITESPARSE_LIBRARIES} ${G2O_LIBRARIES})

catkin_python_setup()

IF(NOT MSVC)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support which is required 
  by linked third party packages starting from ROS Jade. Ignore this message for ROS Indigo.")
endif()
endif()


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  TrajectoryPointMsg.msg
  TrajectoryMsg.msg
  FeedbackMsg.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES geometry_msgs std_msgs costmap_converter sensor_msgs std_srvs
)

#add dynamic reconfigure api
generate_dynamic_reconfigure_options(
  cfg/TebLocalPlannerReconfigure.cfg
  #cfg/TiagoReconfigure.cfg
)



catkin_package(
  INCLUDE_DIRS include ${EXTERNAL_INCLUDE_DIRS}
  LIBRARIES navigation_explainer ${EXTERNAL_LIBS}
  CATKIN_DEPENDS
    base_local_planner
    costmap_2d
    costmap_converter
    dynamic_reconfigure
    geometry_msgs
    interactive_markers
    message_runtime
    nav_core
    nav_msgs
    pluginlib
    roscpp
    mbf_costmap_core
    std_msgs
    tf2
    tf2_ros
    visualization_msgs
    sensor_msgs
    std_srvs
  DEPENDS SUITESPARSE G2O
)

include_directories(include)
include_directories(
  SYSTEM
  ${EXTERNAL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

## Build the navigation_explainer library

add_library(navigation_explainer
   src/tlp/timed_elastic_band.cpp
   src/tlp/optimal_planner.cpp
   src/tlp/obstacles.cpp
   src/tlp/visualization.cpp
   src/tlp/recovery_behaviors.cpp
   src/tlp/teb_config.cpp
   src/tlp/homotopy_class_planner.cpp
   #src/tlp/teb_local_planner_ros.cpp
   src/tlp/graph_search.cpp
   src/tlp/costmap_wrapper.cpp
   src/tlp/my_wrapper.cpp
)

# Dynamic reconfigure: make sure configure headers are built before any node using them
add_dependencies(navigation_explainer ${PROJECT_NAME}_gencfg)
# Generate messages before compiling the lib
add_dependencies(navigation_explainer ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries(navigation_explainer
   ${EXTERNAL_LIBS}
   ${catkin_LIBRARIES}
)

add_executable(pni src/tlp/pni.cpp)


target_link_libraries(pni
   navigation_explainer
   ${EXTERNAL_LIBS}
   ${catkin_LIBRARIES}
)

## Mark executables and/or libraries for installation
install(TARGETS navigation_explainer
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(TARGETS pni
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/tlp/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   #FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
)

install(DIRECTORY
  launch cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)
