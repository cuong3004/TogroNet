#!/bin/bash
set -e

PI3_USER="pi"
PI3_HOST="192.168.50.3"

DEV_WS="/home/pi/dev_ws"

DEV_PKGS=(
  my_robot_bringup
  my_robot_description
  my_robot_hardware
)

echo "=================================="
echo " Deploy selected ROS packages to Pi3"
echo " Target: ${PI3_USER}@${PI3_HOST}"
echo "=================================="

echo "[1] Check SSH..."
ssh ${PI3_USER}@${PI3_HOST} "echo Connected to \$(hostname)"

echo "[2] Create folders on Pi3..."
ssh ${PI3_USER}@${PI3_HOST} "
mkdir -p ${DEV_WS}/src
mkdir -p ${DEV_WS}/build
mkdir -p ${DEV_WS}/install
"

copy_install_setup_files() {
    LOCAL_INSTALL="$1"
    REMOTE_INSTALL="$2"

    echo "Copy top-level setup files from ${LOCAL_INSTALL}"

    shopt -s nullglob

    FILES=(
      "${LOCAL_INSTALL}"/setup.*
      "${LOCAL_INSTALL}"/local_setup.*
      "${LOCAL_INSTALL}"/_local_setup_util*.py
      "${LOCAL_INSTALL}"/.colcon_install_layout
      "${LOCAL_INSTALL}"/COLCON_IGNORE
    )

    if [ ${#FILES[@]} -eq 0 ]; then
        echo "WARNING: no setup files found in ${LOCAL_INSTALL}"
        return
    fi

    rsync -avz "${FILES[@]}" \
      ${PI3_USER}@${PI3_HOST}:${REMOTE_INSTALL}/

    shopt -u nullglob
}

echo "[3] Copy dev_ws top-level install setup files..."
copy_install_setup_files "${DEV_WS}/install" "${DEV_WS}/install"

echo "[4] Copy dev_ws selected src packages..."
for pkg in "${DEV_PKGS[@]}"; do
    echo "  src/${pkg}"
    rsync -avz --delete \
      "${DEV_WS}/src/${pkg}" \
      ${PI3_USER}@${PI3_HOST}:${DEV_WS}/src/
done

echo "[5] Copy dev_ws selected build packages..."
for pkg in "${DEV_PKGS[@]}"; do
    if [ -d "${DEV_WS}/build/${pkg}" ]; then
        echo "  build/${pkg}"
        rsync -avz --delete \
          "${DEV_WS}/build/${pkg}" \
          ${PI3_USER}@${PI3_HOST}:${DEV_WS}/build/
    else
        echo "  WARNING: ${DEV_WS}/build/${pkg} not found, skip"
    fi
done

echo "[6] Copy dev_ws selected install packages..."
for pkg in "${DEV_PKGS[@]}"; do
    if [ -d "${DEV_WS}/install/${pkg}" ]; then
        echo "  install/${pkg}"
        rsync -avz --delete \
          "${DEV_WS}/install/${pkg}" \
          ${PI3_USER}@${PI3_HOST}:${DEV_WS}/install/
    else
        echo "  WARNING: ${DEV_WS}/install/${pkg} not found, skip"
    fi
done

echo "=================================="
echo " Deploy done."
echo "=================================="

echo "Test on Pi3:"
echo "ssh ${PI3_USER}@${PI3_HOST}"
echo "source /opt/ros/jazzy/setup.bash"
echo "source ~/dev_ws/install/setup.bash"
echo "ros2 pkg list | grep my_robot"
echo "ros2 launch my_robot_bringup my_robot.launch.py"
