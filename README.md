Fine-Grained Sneaker Classification Using Pretrained CNNs
=========================================================

This repository presents a 3-phase project on classifying sneakers into 21 distinct classes using transfer learning with pretrained convolutional neural networks (CNNs). The final model is deployed on a robot for real-world sneaker recognition using robot-captured images.

---

Project Phases
--------------

Phase 1: Baseline Training with Transfer Learning
- Implemented feature extraction using InceptionV3 and MobileNetV2 pretrained on ImageNet.
- Only trained a custom classifier head (convolutional base frozen).
- Evaluated performance on a balanced dataset of 21 sneaker models.

Phase 2: Fine-Tuning and Data Augmentation
- Unfroze selected convolutional layers for fine-tuning.
- Applied data augmentation (rotation, flip, zoom, brightness).
- Integrated learning rate scheduling and early stopping.
- Best model: InceptionV3 Variant 2 with 75.20% test accuracy.

Phase 3: Real-World Deployment Using Robot-Captured and Supplemental Images
- Collected 145 robot-captured images for 3 sneaker classes:
  - Nike_Infinity_React_3 (48), Nike_Vomero_17 (50), Nike_Winflo_10 (47)
- Fine-tuned the InceptionV3 model on robot-specific data for 3-class classification.
- Integrated 500 supplemental images (25 images for each of 20 other classes) — contributed by another group — to enrich the training dataset (1 class was missing).
- Deployed the fine-tuned 3-class model on a ROS2-powered robot.
- Mapped detected sneaker classes to real-time robot actions (e.g., move forward, turn left).

---

Dataset Overview
----------------

- 21 fine-grained sneaker classes (e.g., Nike Air Max 90, Adidas Ultraboost).
- 50+ images per class, collected via:
  - Public sources (social media, product listings)
  - Self-captured photographs taken from different angles and lighting conditions
- Folder-structured dataset for easy loading using ImageFolder.

Note: Full dataset is excluded due to copyright. A 500-image supplemental dataset (25 images × 20 classes) was used in Phase 3 to improve training diversity.

---

Results Snapshot
----------------

| Model              | Strategy                  | Dataset            | Accuracy   |
|--------------------|---------------------------|--------------------|------------|
| InceptionV3        | Feature Extraction        | Sneaker Dataset    | 56.30%     |
| MobileNetV2        | Feature Extraction        | Sneaker Dataset    | 57.87%     |
| InceptionV3 (V2)   | Fine-Tuned + Augment      | Sneaker Dataset    | 75.20%     |
| InceptionV3        | Pre-Fine-Tune (21-Class)  | Robot Test Set     | 5.88%      |
| InceptionV3        | Fine-Tuned (3-Class Only) | Robot Test Set     | 82.35%     |

Note: A supplemental dataset (25 images × 20 classes) was contributed by another group during Phase 3. One class was missing, so only 20 of the 21 classes were enhanced. This dataset was used strictly for training enrichment. All other aspects — including model training, robot integration, evaluation, documentation, and deployment — were independently completed by Shaimon Rahman.

---

Setup and Dependencies
----------------------

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Key Libraries:
- TensorFlow / Keras
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- Seaborn

---

Robot Integration
-----------------

Phase 3 showcases real-world deployment:
- Collected 145 robot images using `ros2_image_capture_from_camera.py`, covering 3 classes with ~30–50 images per class under varied lighting conditions.
- Robot used fine-tuned 3-class InceptionV3 model.
- Model deployed for on-device inference and sneaker-triggered robot movement.

---

Repository Structure
--------------------

```
Sneaker-Classification/
├── Phase_1/
│   └── Fine-Tuning_Pretrained_CNNs_Phase_1.ipynb
├── Phase_2/
│   └── Fine-Tuning_Pretrained_CNNs_Phase_2.ipynb
├── Phase_3/
│   └── Fine-Tuning_Pretrained_CNNs_Phase_3.ipynb
│   └── Robot_images/ (optional: sample robot-captured images)
├── robot_scripts/
│   ├── ros2_image_capture_from_camera.py
│   ├── ros2_deploy_inceptionv3_sneaker_detection_3class.py
│   └── README_robot.md
├── requirements.txt
├── dataset_description.md
├── README.md
└── LICENSE
```

---

Author
------

Shaimon Rahman  
Master of Information Technology in Artificial Intelligence  
Macquarie University, Australia

Acknowledgment: A dataset containing 500 images (25 each for 20 of the 21 classes) was contributed by another group during Phase 3. This was used solely to enhance training diversity. The full project — from design and modeling to robot deployment and reporting — was independently completed by Shaimon Rahman.

---
License
-------

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.
