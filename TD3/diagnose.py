#!/usr/bin/env python3
"""Diagnose robot model and TF issues"""

import rospy
from gazebo_msgs.msg import ModelStates

def diagnose():
    rospy.init_node("robot_diagnose")
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. Check model names in Gazebo
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("📋 GAZEBO MODELS")
    print("="*60)
    
    try:
        msg = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5)
        print(f"Found {len(msg.name)} models:")
        for i, name in enumerate(msg.name):
            pos = msg.pose[i].position
            print(f"  {i+1}. '{name}' at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
    except Exception as e:
        print(f"❌ Failed to get models: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. Check TF frames
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🔗 TF FRAMES")
    print("="*60)
    
    import tf
    try:
        listener = tf.TransformListener()
        rospy.sleep(2.0)  # Wait for TF buffer
        
        frames = listener.getFrameStrings()
        print(f"Found {len(frames)} frames:")
        for frame in sorted(frames):
            print(f"  - {frame}")
    except Exception as e:
        print(f"❌ Failed to get TF: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. Check odom -> base_link transform
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("📍 ODOM -> BASE_LINK TRANSFORM")
    print("="*60)
    
    try:
        listener = tf.TransformListener()
        rospy.sleep(1.0)
        
        (trans, rot) = listener.lookupTransform('odom', 'base_link', rospy.Time(0))
        print(f"Position: ({trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f})")
        print(f"Rotation: ({rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f})")
        
        # Check if stuck at origin
        if abs(trans[0]) < 0.01 and abs(trans[1]) < 0.01:
            print("⚠️ WARNING: Transform is at origin! Something is publishing static TF!")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. Check who publishes base_link
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("🔍 TF PUBLISHERS FOR BASE_LINK")
    print("="*60)
    print("Run this command to see who publishes base_link:")
    print("  rosrun tf tf_monitor base_link")
    print("  rosrun tf tf_echo odom base_link")

if __name__ == "__main__":
    diagnose()