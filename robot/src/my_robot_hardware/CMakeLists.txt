cmake_minimum_required(VERSION 3.8)
project(my_robot_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)   # 🔥 THÊM
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)                # 🔥 THÊM
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)

# Boost (asio)
find_package(Boost REQUIRED)



ament_python_install_package(${PROJECT_NAME})  # 🔥 THÊM

# ======================
# Library
# ======================
add_library(
  ${PROJECT_NAME}
  SHARED
  src/mobile_base_hardware_interface.cpp
)

add_executable(twist_relay src/twist_relay.cpp)
ament_target_dependencies(twist_relay rclcpp geometry_msgs)

# Include dirs
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    include
    ${Boost_INCLUDE_DIRS}
)

# ROS deps
ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  rclcpp_lifecycle
  hardware_interface
  pluginlib
)

# ======================
# Plugin
# ======================
pluginlib_export_plugin_description_file(
  hardware_interface
  my_robot_hardware_interface.xml
)

# ======================
# Install
# ======================
install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
)

install(
  DIRECTORY include
  DESTINATION include
)

install(TARGETS
  twist_relay
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY rviz launch
  DESTINATION share/${PROJECT_NAME}/
)

# 🔥 CÀI FILE PYTHON
install(
  PROGRAMS
    my_robot_hardware/mpu6050_driver.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

ament_package()