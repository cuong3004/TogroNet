# pi_cam_teleop

Gói ROS2 kết hợp:
- Stream camera Pi (đọc trực tiếp bằng `Picamera2`, hiển thị bằng OpenCV).
- Điều khiển robot bằng bàn phím giống `teleop_twist_keyboard` — nhưng phím
  được đọc **ngay trên cửa sổ camera** (phải click vào cửa sổ camera để nó
  được focus trước khi gõ phím).
- Phím `c` để chụp ảnh từ khung hình camera hiện tại.

## Cài đặt phụ thuộc (không quản lý qua colcon/rosdep)

```bash
sudo apt install -y python3-opencv python3-picamera2
```

## Build

```bash
cd ~/dev_ws
colcon build --packages-select pi_cam_teleop
source install/setup.bash
```

## Chạy (tương đương lệnh teleop_twist_keyboard bạn đang dùng)

```bash
ros2 run pi_cam_teleop pi_cam_teleop \
  --ros-args \
  -r cmd_vel:=/diff_drive_controller/cmd_vel \
  -p stamped:=true
```

Một cửa sổ camera sẽ hiện lên. Click vào cửa sổ đó rồi gõ phím để điều khiển.

## Bảng phím

```
   u    i    o
   j    k    l
   m    ,    .
```

- `i` / `,` : tiến / lùi
- `j` / `l` : quay trái / phải tại chỗ
- `u` / `o` : tiến + rẽ trái / phải
- `m` / `.` : lùi + rẽ trái / phải
- phím khác (vd `k`) : dừng robot
- `q` / `z` : tăng / giảm cả tốc độ tiến và tốc độ quay
- `w` / `x` : tăng / giảm tốc độ tiến (linear)
- `e` / `r` : tăng / giảm tốc độ quay (angular)
- `c` : **chụp ảnh** từ camera
- `a` : bật lại chế độ **tự động cân bằng trắng** (AWB auto) — dùng khi
  ra ngoài trời / đổi ánh sáng và muốn camera tự căn chỉnh lại màu
- `s` : **lưu & khóa** hệ số cân bằng trắng hiện tại (ứng với ánh sáng lúc
  đó nhìn màu đã đẹp) — ghi vào file, chuyển camera sang chế độ thủ công
  dùng đúng hệ số đó, và những lần chạy sau sẽ tự nạp lại file này
- `ESC` hoặc `Ctrl+C` (ở terminal) : thoát chương trình

> Lưu ý: gói gốc `teleop_twist_keyboard` dùng phím `c` để giảm tốc độ quay.
> Vì `c` ở gói này được dùng để chụp ảnh, chức năng giảm tốc độ quay được
> chuyển sang phím `r`.

## Tham số (dùng `-p ten:=gia_tri`)

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `stamped` | `false` | `true` → publish `TwistStamped`, `false` → `Twist` |
| `frame_id` | `''` | frame_id trong header khi `stamped:=true` |
| `speed` | `0.2` | tốc độ tiến/lùi ban đầu (m/s) |
| `turn` | `1.0` | tốc độ quay ban đầu (rad/s) |
| `publish_rate_hz` | `20.0` | tần số publish `cmd_vel` |
| `camera_width` / `camera_height` | `640` / `480` | độ phân giải camera |
| `camera_fps` | `30.0` | FPS camera |
| `save_dir` | `~/Pictures/pi_cam_teleop` | thư mục lưu ảnh chụp |
| `window_name` | `Pi Cam Teleop` | tên cửa sổ hiển thị |
| `wb_gains_file` | `~/.pi_cam_teleop/wb_gains.json` | file lưu hệ số cân bằng trắng (phím `s`), tự nạp lại mỗi lần chạy nếu tồn tại |

Ví dụ đổi tốc độ và nơi lưu ảnh:

```bash
ros2 run pi_cam_teleop pi_cam_teleop \
  --ros-args \
  -r cmd_vel:=/diff_drive_controller/cmd_vel \
  -p stamped:=true \
  -p speed:=0.15 \
  -p turn:=0.8 \
  -p save_dir:=/home/pi/dev_ws/dataset/captures
```
