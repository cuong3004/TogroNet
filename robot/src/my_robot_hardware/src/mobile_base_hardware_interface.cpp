#include "my_robot_hardware/mobile_base_hardware_interface.hpp"

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <hardware_interface/lexical_casts.hpp>

#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <memory>

namespace mobile_base_hardware
{

hardware_interface::CallbackReturn
MobileBaseHardwareInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{

  // RCLCPP_INFO(get_logger(), "Debug here");
  if (
    hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(get_logger(), "MobileBase on_init()");

  /* ===============================
   * 1. VALIDATE JOINT INTERFACES
   * =============================== */
  for (const auto & joint : info_.joints)
  {
    if (joint.command_interfaces.size() != 1 ||
        joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY)
    {
      RCLCPP_FATAL(
        get_logger(),
        "Joint '%s' must have exactly 1 velocity command interface",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }

    if (joint.state_interfaces.size() != 2 ||
        joint.state_interfaces[0].name != hardware_interface::HW_IF_POSITION ||
        joint.state_interfaces[1].name != hardware_interface::HW_IF_VELOCITY)
    {
      RCLCPP_FATAL(
        get_logger(),
        "Joint '%s' must have state interfaces: [position, velocity] (in order)",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }
  // RCLCPP_INFO(get_logger(), "Debug here");

  /* ===============================
   * 2. VALIDATE IMU SENSOR
   * =============================== */
  bool imu_found = false;

  for (const auto & sensor : info_.sensors)
  {
    if (sensor.name == "imu")
    {
      imu_found = true;

      if (sensor.state_interfaces.size() != 10)
      {
        RCLCPP_FATAL(
          get_logger(),
          "IMU must expose exactly 10 state interfaces");
        return hardware_interface::CallbackReturn::ERROR;
      }

      static const std::vector<std::string> required = {
        "orientation.x", "orientation.y", "orientation.z", "orientation.w",
        "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
        "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"
      };

      for (const auto & iface : required)
      {
        auto it = std::find_if(
          sensor.state_interfaces.begin(),
          sensor.state_interfaces.end(),
          [&](const auto & si) { return si.name == iface; });

        if (it == sensor.state_interfaces.end())
        {
          RCLCPP_FATAL(
            get_logger(),
            "Missing IMU state interface: imu/%s",
            iface.c_str());
          return hardware_interface::CallbackReturn::ERROR;
        }
      }
    }
  }
  // RCLCPP_INFO(get_logger(), "Debug here");

  if (!imu_found)
  {
    RCLCPP_FATAL(get_logger(), "IMU sensor 'imu' not found");
    return hardware_interface::CallbackReturn::ERROR;
  }

  /* ===============================
   * 3. READ PARAMETERS
   * =============================== */
  port_ =  "/dev/ttyACM0";// info_.hardware_parameters.at("port");

  RCLCPP_INFO(get_logger(), "Debug here123");

  // driver_.reset();  // IMPORTANT: no hardware here

  // RCLCPP_INFO(get_logger(), "Debug here4");

  RCLCPP_INFO(get_logger(), "on_init() successful");
  return hardware_interface::CallbackReturn::SUCCESS;
}


/* ============================================================
 * on_configure()
 *  - allocate resource
 * ============================================================ */
hardware_interface::CallbackReturn
MobileBaseHardwareInterface::on_configure(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Configuring MobileBase hardware");

  driver_ = std::make_shared<PicoMotorDriver>(port_);

  if (!driver_->init())
  {
    RCLCPP_ERROR(get_logger(), "Driver init failed");
    return hardware_interface::CallbackReturn::ERROR;
  }

  /* reset all states & commands */
  for (const auto & [name, _] : joint_state_interfaces_)
  {
    set_state(name, 0.0);
  }
  for (const auto & [name, _] : joint_command_interfaces_)
  {
    set_command(name, 0.0);
  }

  set_state("imu/orientation.w", 1.0);

  RCLCPP_INFO(get_logger(), "Configured successfully");
  return hardware_interface::CallbackReturn::SUCCESS;
}

/* ============================================================
 * on_activate()
 * ============================================================ */
hardware_interface::CallbackReturn
MobileBaseHardwareInterface::on_activate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Activating MobileBase hardware");

  if (!driver_->activate())
  {
    RCLCPP_ERROR(get_logger(), "Driver activation failed");
    return hardware_interface::CallbackReturn::ERROR;
  }

  /* command = state on start */
  for (const auto & [name, _] : joint_command_interfaces_)
  {
    set_command(name, get_state(name));
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

/* ============================================================
 * on_deactivate()
 * ============================================================ */
hardware_interface::CallbackReturn
MobileBaseHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Deactivating MobileBase hardware");

  if (driver_)
    driver_->deactivate();

  for (const auto & [name, _] : joint_command_interfaces_)
  {
    set_command(name, 0.0);
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

/* ============================================================
 * read()
 * ============================================================ */
hardware_interface::return_type
MobileBaseHardwareInterface::read(
  const rclcpp::Time &,
  const rclcpp::Duration & period)
{
  auto data = driver_->getVelocityRadPerSec();

  if (data.size() != 12)
  {
    RCLCPP_ERROR(
      get_logger(),
      "Driver returned invalid data size: %zu (expected 8)",
      data.size());

    if (!data.empty())
    {
      std::ostringstream oss;
      oss << "Raw data: [ ";
      for (const auto & v : data)
        oss << v << " ";
      oss << "]";
      RCLCPP_ERROR(get_logger(), "%s", oss.str().c_str());
    }

    return hardware_interface::return_type::ERROR;
  }

  const double v_left  = data[0];
  const double v_right = data[1];

  set_state("base_left_wheel_joint/velocity",  v_left);
  set_state("base_right_wheel_joint/velocity", v_right);

  set_state(
    "base_left_wheel_joint/position",
    get_state("base_left_wheel_joint/position") + v_left * period.seconds());

  set_state(
    "base_right_wheel_joint/position",
    get_state("base_right_wheel_joint/position") + v_right * period.seconds());

  set_state("imu/angular_velocity.x", data[2]);
  set_state("imu/angular_velocity.y", data[3]);
  set_state("imu/angular_velocity.z", data[4]);

  set_state("imu/linear_acceleration.x", data[5]);
  set_state("imu/linear_acceleration.y", data[6]);
  set_state("imu/linear_acceleration.z", data[7]);

  set_state("imu/orientation.x", data[8]);
  set_state("imu/orientation.y", data[9]);
  set_state("imu/orientation.z", data[10]);
  set_state("imu/orientation.w", data[11]);
  return hardware_interface::return_type::OK;
}


/* ============================================================
 * write()
 * ============================================================ */
hardware_interface::return_type
MobileBaseHardwareInterface::write(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  const double vl = get_command("base_left_wheel_joint/velocity");
  const double vr = get_command("base_right_wheel_joint/velocity");

  driver_->setTargetVelocityRadPerSec({vl, vr});
  return hardware_interface::return_type::OK;
}

}  // namespace mobile_base_hardware

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  mobile_base_hardware::MobileBaseHardwareInterface,
  hardware_interface::SystemInterface)