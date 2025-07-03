# Dataset Description

This project uses a fine-grained sneaker image dataset for classification across 21 distinct sneaker classes. The dataset is structured to support image classification tasks using deep learning models and follows the standard folder-based structure compatible with PyTorch’s `ImageFolder` format.

---

## Dataset Overview

- **Total Classes:** 21 fine-grained sneaker models  
- **Images per Class:** 50+ (varies slightly)  
- **Image Types:** JPEG, PNG  
- **Image Sources:**
  - Publicly available product listings, sneaker store pages, and social media posts  
  - Self-captured photos using mobile phones  
  - Robot-captured images (used only for 3 classes in Phase 3)

---

## Robot-Captured Images (Phase 3)

- **Classes Covered:** 
  - Nike_Infinity_React_3 (48 images total: 34 train / 14 test)  
  - Nike_Vomero_17 (50 images total: 35 train / 15 test)  
  - Nike_Winflo_10 (47 images total: 33 train / 14 test)  
- **Total Robot Images:** 145  
- **Capture Device:** ROS2-powered RGB camera  
- **Use Case:** Used for fine-tuning the model to perform real-world classification directly from a robot  
- **Conditions:** Captured under varied lighting conditions to simulate real-world deployment scenarios

---

## Folder Structure

The dataset follows this folder structure:

```
dataset/
├── Nike_Air_Max_90/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Adidas_Ultraboost/
│   └── ...
└── ...
```

---

## Data Availability

Due to copyright and privacy considerations, the full dataset is not included in this repository.

- A small subset or synthetic example may be shared upon request.  
- You are encouraged to recreate your own dataset using similar methods described in the notebooks and scripts.

---

## Licensing

Only self-captured and robot-captured images are eligible for redistribution under fair use. Publicly sourced images should not be redistributed without permission from the original copyright holders.

---

## Contact

For dataset structure questions or usage inquiries, contact: **Shaimon Rahman**