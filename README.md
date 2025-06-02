# 3D U-Net for Maxilla Segmentation
This repository contains PyTorch code for segmenting medical images using a custom 3D U-Net (U3Net).
## Usage
1. Place your `.nrrd` files into `data/maxilla/`.
2. Update `config.yaml` with the correct shape and paths.
3. Train the model:
```bash
python train.py
