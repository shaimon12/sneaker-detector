Robot Integration Guide
=======================

This directory contains scripts used to collect images with a ROS2-powered robot and to deploy the trained sneaker classification model for real-time inference and motion control.

---

Components
----------

1. `ros2_image_capture_from_camera.py`
- Captures images from the robot's RGB camera.
- Saves images to the `robot_dataset/` folder in a structured format.
- Publishes from `/depth_cam/rgb/image_raw`.

2. `ros2_deploy_inceptionv3_sneaker_detection_3class.py`
- Loads the fine-tuned 3-class InceptionV3 model.
- Subscribes to the RGB camera topic and predicts sneaker class.
- Maps predictions to ROS2 Twist messages for robot movement via `/cmd_vel`.

---

Setup Instructions
------------------

```bash
# Source your ROS2 environment
source /opt/ros/foxy/setup.bash

# Run image capture node
ros2 run sneaker_robot ros2_image_capture_from_camera.py

# Run deployment node
ros2 run sneaker_robot ros2_deploy_inceptionv3_sneaker_detection_3class.py
```

---

Sneaker Class to Robot Action Mapping
-------------------------------------

| Sneaker Class             | Action       |
|---------------------------|--------------|
| Nike_Infinity_React_3     | Move Forward |
| Nike_Vomero_17            | Turn Left    |
| Nike_Winflo_10            | Turn Right   |

---

Model File
----------

Download the fine-tuned model here:  
[inceptionv3_robot_final_pytorch_3_unfrozen.pth](<your-cloud-link-here>)

After downloading, place it in the following directory on your robot:

```
/home/ubuntu/inceptionv3_robot_final_pytorch_3_unfrozen.pth
```

Update the path in your deployment script if necessary.

---

Robot Dataset Info
-------------------

- Total robot-captured images: 145
- Covered Classes:
  - Nike_Infinity_React_3: 48 images (34 train / 14 test)
  - Nike_Vomero_17: 50 images (35 train / 15 test)
  - Nike_Winflo_10: 47 images (33 train / 14 test)
- Captured under varying lighting conditions using the onboard RGB camera
- Used for real-world validation of the 3-class sneaker detection model

---

Author
------

Developed and maintained by **Shaimon Rahman**