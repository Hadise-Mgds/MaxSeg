# MaxSeg

### CBCT Dataset for Deep Learning-based Segmentation and Virtual Surgical Planning in Bimaxillary Surgery
  
# Abstract

Virtual Surgical Planning (VSP) is essential for patient-centric care in craniomaxillofacial surgeries, particularly in bimaxillary orthognathic surgeries involving maxilla and mandible repositioning. VSP in bimaxillary orthognathic surgery requires 3D surface models of the target structure that rely on accurate segmentation from imaging data, such as Computed Tomography (CT) or Cone-Beam Computed Tomography (CBCT) scans. Accurate maxilla segmentation is vital yet challenging due to its complex morphology compared to the mandible. While manual maxilla segmentation remains common, it is time-consuming, laborious, and lacks reproducibility. To address these limitations, automating maxilla segmentation using Deep Learning methods is proposed. However, it's important to acknowledge that manual segmentation is an important step as the gold standard for validating purposes in supervised deep learning methods. To the best of our knowledge, there are no publicly available maxilla-labeled datasets. In this study, we present the first open-access high-quality collection of CBCT images and corresponding maxilla labels from patients who underwent orthognathic surgery called MaxSeg.

# Keywords:

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

# Technical Validation

The main contribution of this paper is to provide a valid and reliable dataset for automatic maxillary bone segmentation. Hence, our focus went on proving that our data illustrates the proficient level of performance to achieve this goal. After the segmentation masks for 27 subjects were prepared, an automated 3D segmentation model was trained. In this study, we implemented two-stage coarse and fine segmentation to segment the maxilla. In both stages, we employed the 3D-UNet architecture, which demonstrated effectiveness in segmenting bony structures. The purpose of this two-stage segmentation approach was to enable hierarchical feature representation. During the coarse stage, the general structure and shape of the maxilla were extracted. Subsequently, in the fine stage, it focused on capturing finer details and intricacies, including the four borders of the frontal, zygomatic, alveolar, and palatine bones, as well as the inner wall of the sinuses. 

![image](https://github.com/user-attachments/assets/12858771-6d1f-47cc-b812-c29c6fb08ece)

3D-UNet architecture

All CBCT volumes in the MaxSeg have different resolutions with a varying number of slices and large sizes. Therefore, it is important to consider the adaptation when training on 3D images. To this end, we used the patch technique in our coarse and fine segmentation. Initially, the coarse step extracted the overall structure and shape of the maxilla. Following this, the fine step focused on capturing detailed features, such as the sutures of the frontal, zygomatic, alveolar, and palatine bones, as well as the inner wall of the sinuses. 

![image](https://github.com/user-attachments/assets/c5418e54-9eea-4e90-bb87-13d45cac7e21)

The process of data preparation for coarse and fine segmentation
 
# The segmentation results:

The coarse segmentation method was able to effectively represent the overall structure and form of the maxilla. Consequently, the results of the fine segmentation delineate the areas of low bone density and the location of dental artifacts.

![image](https://github.com/user-attachments/assets/bbc643fe-174e-46cb-be13-82db92a48d7e)

Qualitative results of coarse segmentation of the maxilla.

![image](https://github.com/user-attachments/assets/09e0a35f-4f61-452d-9285-8573fb10596c)

Results of fine segmentation in the corresponding areas missed by coarse segmentation of the maxilla.


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

