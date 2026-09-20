#ifndef MOBILE_BASE_HARDWARE_INTERFACE_HPP
#define MOBILE_BASE_HARDWARE_INTERFACE_HPP

#include <rclcpp/rclcpp.hpp>
#include "hardware_interface/system_interface.hpp"
#include "my_robot_hardware/pico_motor_driver.hpp"

namespace mobile_base_hardware
{

class MobileBaseHardwareInterface
  : public hardware_interface::SystemInterface
{
public:

  // hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  
  // // std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  // // std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  // // Lifecycle node override
  // hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  // hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  // hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  // SystemInterface override

  // hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  // hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

    hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::shared_ptr<PicoMotorDriver> driver_;

  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_imu_;
  std::vector<double> hw_commands_;
  // int left_motor_id_;
  // int right_motor_id_;
  std::string port_;
};

}  // namespace mobile_base_hardware

#endif  // MOBILE_BASE_HARDWARE_INTERFACE_HPP