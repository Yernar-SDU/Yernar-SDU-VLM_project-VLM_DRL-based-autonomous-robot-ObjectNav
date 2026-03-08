#!/bin/bash
echo "==========================================="
echo "   🔍 ROS ROBOT DIAGNOSTIC SUITE 🔍"
echo "==========================================="

echo -e "\n1. CHECKING TF CONNECTIVITY (Is the robot falling apart?)"
# Check if odom can reach base_link
timeout 2s rosrun tf tf_echo odom base_link || echo "❌ ERROR: odom -> base_link link is BROKEN!"

echo -e "\n2. CHECKING SENSOR ALIGNMENT"
# Check the frame_id of the laser scan
LASER_FRAME=$(rostopic echo /r1/front_laser/scan -n 1 | grep frame_id | head -1)
echo "Laser is publishing on frame: $LASER_FRAME"
if [[ $LASER_FRAME == *"chassis"* || $LASER_FRAME == *"base_link"* ]]; then
    echo "✅ Laser frame looks connected to robot."
else
    echo "⚠️ WARNING: Laser frame might be disconnected from robot hierarchy."
fi

echo -e "\n3. CHECKING NAVIGATION STATUS (Why is it spinning?)"
# Check if move_base is in recovery mode
NAV_STATUS=$(rostopic echo /move_base/status -n 1 | grep "text" | tail -1)
echo "Latest move_base status: $NAV_STATUS"

echo -e "\n4. DETECTING DUPLICATE TF PUBLISHERS"
# Check if multiple nodes are fighting for the same link
rosnode list | grep -E "static_transform|odom_to_base|map_to_odom"

echo -e "\n5. SCAN TOPIC REMAPPING CHECK"
# Verify move_base is actually listening to the r1 laser
rostopic info /r1/front_laser/scan | grep "/move_base" || echo "❌ ERROR: move_base is NOT listening to /r1/front_laser/scan!"

echo -e "\n6. ODOM TOPIC REMAPPING CHECK"
rostopic info /r1/odom_custom | grep "/move_base" || echo "❌ ERROR: move_base is NOT listening to /r1/odom_custom!"

echo -e "\n==========================================="