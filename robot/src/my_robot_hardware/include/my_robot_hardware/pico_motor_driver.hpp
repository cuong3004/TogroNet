#ifndef PICO_MOTOR_DRIVER_HPP
#define PICO_MOTOR_DRIVER_HPP

#include <boost/asio.hpp>
#include <boost/system/error_code.hpp>

#include <rclcpp/rclcpp.hpp>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <algorithm>

// ============================================================
// Protocol
// ============================================================

static constexpr uint8_t USB_PKT_HEADER = 0xAA;
static constexpr uint8_t CMD_GET_VEL_IMU = 0x30;

static constexpr std::size_t MOTOR_COUNT = 2;
static constexpr std::size_t TELEMETRY_VALUE_COUNT = 12;

#pragma pack(push, 1)

struct usb_vel_imu_pkt_t
{
  float vel_l;
  float vel_r;

  float ax;
  float ay;
  float az;

  float gx;
  float gy;
  float gz;

  float qx;
  float qy;
  float qz;
  float qw;
};

#pragma pack(pop)

static_assert(
  sizeof(usb_vel_imu_pkt_t) == 48,
  "usb_vel_imu_pkt_t must contain exactly 48 bytes"
);


// ============================================================
// PicoMotorDriver
// ============================================================

class PicoMotorDriver
{
public:
  PicoMotorDriver(
    const std::string &device,
    int baudrate = 115200)
  : device_(device),
    baudrate_(baudrate),
    io_(),
    serial_(io_)
  {
  }

  ~PicoMotorDriver()
  {
    deactivate();
  }


  // ==========================================================
  // Lifecycle
  // ==========================================================

  bool init()
  {
    RCLCPP_INFO(
      logger(),
      "Init driver for %s at %d baud",
      device_.c_str(),
      baudrate_);

    return true;
  }


  bool activate()
  {
    if (connected_.load())
    {
      return true;
    }

    try
    {
      openSerial();

      connected_.store(true);
      data_valid_.store(false);
      rx_running_.store(true);

      /*
       * Chỉ xóa dữ liệu cũ trước khi bắt đầu RX.
       * Không được flush sau khi telemetry đang chạy.
       */
      flushInput();

      rx_thread_ = std::thread(
        &PicoMotorDriver::serialRxThread,
        this);

      /*
       * RX thread có thể bỏ qua phản hồi ASCII và tìm header 0xAA.
       */
      sendCommand("TELEMETRY_ON\n");
      sleep_ms(1000);

      sendCommand("ENABLE\n");
      sleep_ms(1000);

      setTargetVelocityRadPerSec({0.0, 0.0});

      /*
       * Không báo activate thành công khi chưa nhận được packet.
       * Điều này tránh lần read() đầu tiên trả vector rỗng.
       */
      if (!waitForFirstPacket(std::chrono::milliseconds(3000)))
      {
        RCLCPP_ERROR(
          logger(),
          "No valid telemetry packet received from Pico within 3 seconds");

        deactivate();
        return false;
      }

      RCLCPP_INFO(
        logger(),
        "Driver activated; valid telemetry received");

      return true;
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(
        logger(),
        "Activate failed: %s",
        e.what());

      deactivate();
      return false;
    }
  }


  void deactivate()
  {
    /*
     * Nếu chưa mở serial và không có RX thread thì không làm gì.
     */
    if (!connected_.load() && !rx_thread_.joinable())
    {
      return;
    }

    /*
     * Dừng động cơ trước khi đóng cổng.
     */
    if (connected_.load() && serial_.is_open())
    {
      try
      {
        setTargetVelocityRadPerSec({0.0, 0.0});
        sleep_ms(1000);

        sendCommand("DISABLE\n");
        sleep_ms(1000);

        sendCommand("TELEMETRY_OFF\n");
        sleep_ms(1000);
      }
      catch (const std::exception &e)
      {
        RCLCPP_WARN(
          logger(),
          "Could not send shutdown commands: %s",
          e.what());
      }
    }

    /*
     * Báo RX thread dừng trước.
     */
    rx_running_.store(false);

    /*
     * Hủy boost::asio::read() đang block.
     */
    if (serial_.is_open())
    {
      boost::system::error_code error;

      serial_.cancel(error);

      if (error)
      {
        RCLCPP_DEBUG(
          logger(),
          "serial cancel: %s",
          error.message().c_str());
      }

      serial_.close(error);

      if (error)
      {
        RCLCPP_WARN(
          logger(),
          "serial close failed: %s",
          error.message().c_str());
      }
    }

    if (rx_thread_.joinable())
    {
      rx_thread_.join();
    }

    connected_.store(false);
    data_valid_.store(false);

    RCLCPP_INFO(logger(), "Driver deactivated");
  }


  // ==========================================================
  // Motor commands
  // ==========================================================

  void setTargetVelocityRadPerSec(
    const std::vector<double> &velocities)
  {
    if (velocities.size() != MOTOR_COUNT)
    {
      throw std::runtime_error(
        "setTargetVelocityRadPerSec expects exactly 2 motors");
    }

    checkConnection();

    const std::string command =
      formatVelocity(velocities[0]) + "," +
      formatVelocity(velocities[1]) + "\n";

    sendCommand(command);
  }


  void sendPID(
    const std::string &side,
    double kp,
    double ki,
    double kd)
  {
    char buffer[96];

    std::snprintf(
      buffer,
      sizeof(buffer),
      "SET_PID_%s,%.6f,%.6f,%.6f\n",
      side.c_str(),
      kp,
      ki,
      kd);

    sendCommand(buffer);
  }


  // ==========================================================
  // Feedback
  // ==========================================================

  /*
   * Giữ tên hàm này để không phải sửa nhiều code bên ngoài.
   *
   * Thứ tự dữ liệu trả về:
   *
   *  0: vel_l
   *  1: vel_r
   *  2: gx
   *  3: gy
   *  4: gz
   *  5: ax
   *  6: ay
   *  7: az
   *  8: qx
   *  9: qy
   * 10: qz
   * 11: qw
   */
  std::vector<double> getVelocityRadPerSec()
  {
    if (!connected_.load() || !data_valid_.load())
    {
      return {};
    }

    VelImuData data;

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      data = latest_data_;
    }

    return {
      data.vel_l,
      data.vel_r,

      data.gx,
      data.gy,
      data.gz,

      data.ax,
      data.ay,
      data.az,

      data.qx,
      data.qy,
      data.qz,
      data.qw
    };
  }


  bool hasValidData() const
  {
    return connected_.load() && data_valid_.load();
  }


  bool isConnected() const
  {
    return connected_.load();
  }


private:
  struct VelImuData
  {
    double vel_l{0.0};
    double vel_r{0.0};

    double ax{0.0};
    double ay{0.0};
    double az{0.0};

    double gx{0.0};
    double gy{0.0};
    double gz{0.0};

    double qx{0.0};
    double qy{0.0};
    double qz{0.0};
    double qw{1.0};
  };


  // ==========================================================
  // RX thread
  // ==========================================================

  void serialRxThread()
  {
    RCLCPP_INFO(logger(), "Serial RX thread started");

    while (rx_running_.load())
    {
      try
      {
        /*
         * Tìm byte bắt đầu frame.
         *
         * Cách này cũng tự động bỏ qua phản hồi ASCII như:
         * OK
         * ENABLED
         */
        uint8_t header = 0;

        do
        {
          readExact(&header, 1);

          if (!rx_running_.load())
          {
            break;
          }
        }
        while (header != USB_PKT_HEADER);

        if (!rx_running_.load())
        {
          break;
        }

        uint8_t command = 0;
        uint8_t length = 0;

        readExact(&command, 1);
        readExact(&length, 1);

        /*
         * Chỉ xử lý đúng telemetry packet.
         *
         * Nếu command hoặc length sai, quay lại tìm header mới.
         */
        if (
          command != CMD_GET_VEL_IMU ||
          length != sizeof(usb_vel_imu_pkt_t))
        {
          invalid_header_count_.fetch_add(1);
          continue;
        }

        usb_vel_imu_pkt_t packet{};

        readExact(
          reinterpret_cast<uint8_t *>(&packet),
          sizeof(packet));

        uint8_t received_checksum = 0;
        readExact(&received_checksum, 1);

        const uint8_t calculated_checksum =
          calculateChecksum(command, length, packet);

        if (calculated_checksum != received_checksum)
        {
          checksum_error_count_.fetch_add(1);
          continue;
        }

        if (!validatePacket(packet))
        {
          invalid_packet_count_.fetch_add(1);
          continue;
        }

        {
          std::lock_guard<std::mutex> lock(data_mutex_);

          latest_data_.vel_l = packet.vel_l;
          latest_data_.vel_r = packet.vel_r;

          latest_data_.ax = packet.ax;
          latest_data_.ay = packet.ay;
          latest_data_.az = packet.az;

          latest_data_.gx = packet.gx;
          latest_data_.gy = packet.gy;
          latest_data_.gz = packet.gz;

          latest_data_.qx = packet.qx;
          latest_data_.qy = packet.qy;
          latest_data_.qz = packet.qz;
          latest_data_.qw = packet.qw;
        }

        data_valid_.store(true);
        valid_packet_count_.fetch_add(1);

        data_condition_.notify_all();
      }
      catch (const boost::system::system_error &e)
      {
        /*
         * Khi deactivate gọi cancel/close, readExact sẽ phát sinh
         * operation_aborted hoặc bad descriptor. Đó là dừng bình thường.
         */
        if (!rx_running_.load())
        {
          break;
        }

        RCLCPP_ERROR(
          logger(),
          "Serial RX error: %s",
          e.what());

        break;
      }
      catch (const std::exception &e)
      {
        if (!rx_running_.load())
        {
          break;
        }

        RCLCPP_ERROR(
          logger(),
          "RX thread error: %s",
          e.what());

        break;
      }
    }

    RCLCPP_INFO(
      logger(),
      "Serial RX thread stopped. Valid=%zu, checksum errors=%zu, "
      "invalid packets=%zu, invalid headers=%zu",
      valid_packet_count_.load(),
      checksum_error_count_.load(),
      invalid_packet_count_.load(),
      invalid_header_count_.load());
  }


  // ==========================================================
  // Packet validation
  // ==========================================================

  static uint8_t calculateChecksum(
    uint8_t command,
    uint8_t length,
    const usb_vel_imu_pkt_t &packet)
  {
    uint8_t checksum = command ^ length;

    const auto *bytes =
      reinterpret_cast<const uint8_t *>(&packet);

    for (std::size_t i = 0; i < sizeof(packet); ++i)
    {
      checksum ^= bytes[i];
    }

    return checksum;
  }


  static bool validatePacket(
    const usb_vel_imu_pkt_t &packet)
  {
    const std::array<float, TELEMETRY_VALUE_COUNT> values = {
      packet.vel_l,
      packet.vel_r,

      packet.ax,
      packet.ay,
      packet.az,

      packet.gx,
      packet.gy,
      packet.gz,

      packet.qx,
      packet.qy,
      packet.qz,
      packet.qw
    };

    for (const float value : values)
    {
      if (!std::isfinite(value))
      {
        return false;
      }
    }

    const double quaternion_norm = std::sqrt(
      static_cast<double>(packet.qx) * packet.qx +
      static_cast<double>(packet.qy) * packet.qy +
      static_cast<double>(packet.qz) * packet.qz +
      static_cast<double>(packet.qw) * packet.qw);

    /*
     * Quaternion DMP bình thường có norm gần 1.
     * Khoảng này đủ rộng để phát hiện frame lệch byte.
     */
    if (
      quaternion_norm < 0.5 ||
      quaternion_norm > 1.5)
    {
      return false;
    }

    return true;
  }


  // ==========================================================
  // Serial helpers
  // ==========================================================

  void openSerial()
  {
    if (serial_.is_open())
    {
      return;
    }

    serial_.open(device_);

    serial_.set_option(
      boost::asio::serial_port_base::baud_rate(
        baudrate_));

    serial_.set_option(
      boost::asio::serial_port_base::character_size(
        8));

    serial_.set_option(
      boost::asio::serial_port_base::parity(
        boost::asio::serial_port_base::parity::none));

    serial_.set_option(
      boost::asio::serial_port_base::stop_bits(
        boost::asio::serial_port_base::stop_bits::one));

    serial_.set_option(
      boost::asio::serial_port_base::flow_control(
        boost::asio::serial_port_base::flow_control::none));

    RCLCPP_INFO(
      logger(),
      "Opened serial port %s",
      device_.c_str());
  }


  void sendCommand(const std::string &command)
  {
    checkConnection();

    std::lock_guard<std::mutex> lock(tx_mutex_);

    boost::asio::write(
      serial_,
      boost::asio::buffer(
        command.data(),
        command.size()));
  }


  /*
   * Doc dung "size" byte, nhung khong block vo han.
   *
   * boost::asio::read() dong bo se cho du lieu vinh vien neu Pico
   * ngung gui (vi du sau khi da gui TELEMETRY_OFF luc deactivate()).
   * cancel()/close() tu thread khac khong dam bao danh thuc duoc
   * mot syscall read() dong bo dang block, nen rx_thread_.join() se
   * treo mai va bat buoc phai SIGKILL. Dung poll() voi timeout ngan
   * de vong lap co co hoi kiem tra lai rx_running_ thuong xuyen.
   */
  void readExact(
    uint8_t *data,
    std::size_t size)
  {
    std::size_t total_read = 0;
    const int fd = serial_.native_handle();

    while (total_read < size)
    {
      if (!rx_running_.load())
      {
        throw std::runtime_error("readExact: rx thread is stopping");
      }

      pollfd pfd{};
      pfd.fd = fd;
      pfd.events = POLLIN;

      const int poll_result = poll(&pfd, 1, 200);

      if (poll_result < 0)
      {
        if (errno == EINTR)
        {
          continue;
        }

        throw std::runtime_error(
          std::string("readExact: poll() failed: ") + std::strerror(errno));
      }

      if (poll_result == 0)
      {
        // Timeout, chua co du lieu -> quay lai kiem tra rx_running_.
        continue;
      }

      const ssize_t n = ::read(fd, data + total_read, size - total_read);

      if (n < 0)
      {
        if (errno == EINTR)
        {
          continue;
        }

        throw std::runtime_error(
          std::string("readExact: read() failed: ") + std::strerror(errno));
      }

      if (n == 0)
      {
        throw std::runtime_error("readExact: serial port closed (EOF)");
      }

      total_read += static_cast<std::size_t>(n);
    }
  }


  void flushInput()
  {
    if (!serial_.is_open())
    {
      return;
    }

    const int file_descriptor =
      serial_.native_handle();

    /*
     * Chỉ xóa RX cũ.
     *
     * Không dùng TCIOFLUSH vì có thể xóa luôn command TX.
     */
    if (tcflush(file_descriptor, TCIFLUSH) != 0)
    {
      RCLCPP_WARN(
        logger(),
        "Could not flush serial input buffer");
    }
  }


  bool waitForFirstPacket(
    std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(
      first_packet_mutex_);

    return data_condition_.wait_for(
      lock,
      timeout,
      [this]()
      {
        return data_valid_.load() ||
               !rx_running_.load();
      }) && data_valid_.load();
  }


  static std::string formatVelocity(double velocity)
  {
    /*
     * Giới hạn để tránh tràn buffer format.
     * Có thể đổi giới hạn theo robot thực tế.
     */
    constexpr double MAX_ABS_VELOCITY = 99.99;

    if (!std::isfinite(velocity))
    {
      velocity = 0.0;
    }

    velocity = std::max(
      -MAX_ABS_VELOCITY,
      std::min(MAX_ABS_VELOCITY, velocity));

    char buffer[16];

    const char sign =
      velocity >= 0.0 ? 'p' : 'n';

    std::snprintf(
      buffer,
      sizeof(buffer),
      "%c%05.2f",
      sign,
      std::abs(velocity));

    return std::string(buffer);
  }


  static void sleep_ms(int milliseconds)
  {
    std::this_thread::sleep_for(
      std::chrono::milliseconds(milliseconds));
  }


  void checkConnection() const
  {
    if (!connected_.load() || !serial_.is_open())
    {
      throw std::runtime_error(
        "Pico serial connection is not open");
    }
  }


  static rclcpp::Logger logger()
  {
    return rclcpp::get_logger(
      "PicoMotorDriver");
  }


private:
  std::string device_;
  int baudrate_;

  boost::asio::io_service io_;
  boost::asio::serial_port serial_;

  std::thread rx_thread_;

  std::atomic<bool> connected_{false};
  std::atomic<bool> rx_running_{false};
  std::atomic<bool> data_valid_{false};

  std::mutex tx_mutex_;
  std::mutex data_mutex_;

  std::mutex first_packet_mutex_;
  std::condition_variable data_condition_;

  VelImuData latest_data_{};

  std::atomic<std::size_t> valid_packet_count_{0};
  std::atomic<std::size_t> checksum_error_count_{0};
  std::atomic<std::size_t> invalid_packet_count_{0};
  std::atomic<std::size_t> invalid_header_count_{0};
};

#endif  // PICO_MOTOR_DRIVER_HPP