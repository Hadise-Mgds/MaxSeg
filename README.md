# MaxSeg

## CBCT Dataset for Deep Learning-based Segmentation and Virtual Surgical Planning in Bimaxillary Surgery

# Abstract

Virtual Surgical Planning (VSP) is essential for patient-centric care in craniomaxillofacial surgeries, particularly in bimaxillary orthognathic surgeries involving maxilla and mandible repositioning. VSP in bimaxillary orthognathic surgery requires 3D surface models of the target structure that rely on accurate segmentation from imaging data, such as Computed Tomography (CT) or Cone-Beam Computed Tomography (CBCT) scans. Accurate maxilla segmentation is vital yet challenging due to its complex morphology compared to the mandible. While manual maxilla segmentation remains common, it is time-consuming, laborious, and lacks reproducibility. To address these limitations, automating maxilla segmentation using Deep Learning methods is proposed. However, it's important to acknowledge that manual segmentation is an important step as the gold standard for validating purposes in supervised deep learning methods. To the best of our knowledge, there are no publicly available maxilla-labeled datasets. In this study, we present the first open-access high-quality collection of CBCT images and corresponding maxilla labels from patients who underwent orthognathic surgery called MaxSeg.






* CBCT
* Maxill Segmentation
* Virtual Surgical Planning
* Orthognathic Surgeries

# CBCT artifacts caused by orthodontic brackets in different views: 

![image](https://github.com/user-attachments/assets/33a644f5-9c89-4e65-bec2-6c1f246a557f)

Different views of CBCT scan, labeled as Orthodontics, Sagittal, Coronal, and Axial. Each view highlights artifacts caused by orthodontic brackets, marked with a red circle or lines. 

# Critical blurred areas in coronal view:

![image](https://github.com/user-attachments/assets/3d0c4655-2700-4a09-97eb-a484615fa430)

(a) zygomaticomaxillary suture, (b) frontomaxillary suture (upper arrow) and nasomaxillary suture (lower arrow). Red circles show magnified blurred regions.

# 6 steps of the manual segmentation proposed workflow in Mimics 21.0 to create reliable 3D models of the maxilla:

* Step 1: Import CBCT images and apply semi-automatic thresholding to create initial bone/teeth masks while reducing metal artifacts.
* Step 2: Refine masks manually across axial/coronal/sagittal views using livewire tools to eliminate over-/under-segmentation.
* Step 3: Combine bone and teeth masks via Boolean operations to generate a unified maxilla complex.
* Step 4: Apply smoothing and wrapping algorithms to optimize mask surface topology.
* Step 5: Convert the finalized mask into a 3D object using the "calculate part" function.
* Step 6: Export anonymized maxilla as STL files and convert STL/DICOM to NRRD format for neural network training.

![image](https://github.com/user-attachments/assets/c87c5035-5d9a-408b-b7ff-f80233a1f7e3)

# Results of manual segmentation in critical areas of the maxilla: 
![image](https://github.com/user-attachments/assets/170a24b0-b368-457b-bf8e-28c2adb724c2)

(a) Teeth, (b) Zygomaticomaxillary suture, (c) Anterior nasal spine, (d) Nasomaxillary (vertical) and Frontomaxillary (horizontal) sutures, and (e) Sinus thin walls.
# How to cite
If you use the content of this repository, please consider citing us as below,

@article{Mgds2025open,
title={CBCT Dataset for Deep Learning-based Segmentation and Virtual Surgical Planning in Bimaxillary Surgery},
author={Hadis Moghaddasi, Reza Naghne, Ebrahim Najafzadeh, Alireza Ahmadian, Parastoo Farnia, Alireza parhiz},
journal={Scientific data},

volume={ },

number={ },

pages={ },

year={2025},

publisher {Nature Publishing Group UK London}

}

# project structure
maxilla-segmentation/

├── README.md

├── requirements.txt

├── config.yaml

├── data/

│   └── maxilla/

├── models/

│   └── u3net.py

├── utils/

│   └── dataset.py

├── train.py

└── visualize.py

# 3D U-Net for Maxilla Segmentation
This repository contains PyTorch code for segmenting medical images using a custom 3D U-Net (U3Net).
## Usage
1. Place your `.nrrd` files into `data/maxilla/`.
2. Update `config.yaml` with the correct shape and paths.
3. Train the model:
```bash
python train.py

