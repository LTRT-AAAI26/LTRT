# LTRT
# LTRT: Learnable Truncated Riesz Transform for Low-Light Enhancement and Beyond

Official PyTorch implementation of our AAAI 2026 submission.
LTRT is a novel frequency-aware plugin module designed to improve segmentation accuracy by capturing edge and directional information through a truncated Riesz transform. It can be seamlessly inserted into existing segmentation models such as UNet, ResUNet, and ResUNet++.

# Highlights:

1、Plug-and-Play Design: Integrates easily into mainstream segmentation backbones.

2、Frequency-Aware Edge Enhancement: Models directional and structural cues with low overhead.

![main_light](https://github.com/user-attachments/assets/c9ebb85b-653e-4af3-a410-ee280a7e5fa9)

# Inference & Reproduction Steps on LOL-v1 Dataset

This section demonstrates the effectiveness of the proposed **LTRT plugin** when integrated into existing architectures. By comparing the baseline UNet and our LTRT+UNet, we show that LTRT improves both quantitative (PSNR/SSIM) and qualitative (visual clarity and structure) performance on the LOL-v1 dataset. To reproduce the experimental results of LTRT + UNet on the LOL-v1 dataset, follow the steps below.

1. **Directory Structure**

Ensure the following files are present in the root directory:

── `LOL_v1_best_ltrb_unet.pth`                  # Pretrained model on LOL-v1

── `LOL_v1_LTRB_unet_mid.py`                    # Main code for inference on LOL-v1

── `LOL_v1_test_LTRB_unet.py`                   # Script to evaluate performance (PSNR, SSIM)

── datasets/LOL_v1/                           # LOL-v1 dataset directory. Dataset tip: You can download LOL-v1 from

2. **Run Inference**

Use the following command to run the inference on the LOL-v1 test set using LTRT-UNet and the pretrained weights:

python `LOL_v1_LTRB_unet_mid.py` \

  --weights `LOL_v1_best_ltrb_unet.pth` \
  
  --data_dir ./datasets/LOL_v1/Test/Low \
  
  --save_dir ./results/

LOL_v1 data is available at:

Anonymous Link: https://pan.baidu.com/s/1F19nI5bG_BinyXJ0NtojGw?pwd=pfv2 Code: pfv2 


3. **Evaluate Results**

To calculate PSNR and SSIM between the enhanced images and the ground truth:

python `LOL_v1_test_LTRB_unet.py` \

  --pred_dir ./results/ \
  
  --gt_dir ./datasets/LOL_v1/Test/Normal/

## Quantitative Results on LOL-v1 Dataset

| Model        | PSNR ↑ | SSIM ↑ |
|--------------|--------|--------|
| UNet     | 20.45  | 0.85   |
| **LTRT+UNet** | **23.21**  | **0.88**   |

# Reproduction Instructions: LTRT+UNet on LOL-v2 Real

This repository provides code and pretrained weights to reproduce the experimental results of **LTRT+UNet** on the **LOL-v2 Real** dataset.

## Files

LOL_v2_Real data is available at:

Anonymous Link: https://pan.baidu.com/s/1ra1csxKGiWGFVukE0js0Fw?pwd=b6jc Code: b6jc

- `LOL_v2_best_ltrb_unet.pth`  
  → Pretrained model weights trained on the LOL-v1 dataset and used to test on LOL-v2 Real.

- `LOL_v2_LTRB_unet_mid.py`  
  → Main script to run **LTRT+UNet** on LOL-v2 Real dataset using the pretrained weights.

- `LOL_v2_test_LTRB_unet.py`  
  → Script to evaluate and visualize results for reproducibility testing.

## How to Run

1. **Prepare the LOL-v2 Real Dataset**

   Ensure you have the dataset downloaded and structured like this:

2. **Download Pretrained Model**

Place `LOL_v2_best_ltrb_unet.pth` in the root directory or specify the correct path in the script.

3. **Run Inference**

`python LOL_v2_LTRB_unet_mid.py`

4. **Evaluate Reproducibility**
 
This script evaluates the predicted results using PSNR and SSIM and optionally saves visual comparison outputs.

## Quantitative Results on LOL-v2 Real Dataset

| Model        | PSNR ↑ | SSIM ↑ |
|--------------|--------|--------|
| UNet     | 20.15  | 0.86   |
| **LTRT+UNet** | **22.04**  | **0.91**   |

# Comparative Reproduction: LTRT+UNet vs. SKF+UNet on LOL-v1

This repository provides the code and pretrained weights to reproduce and compare the performance of our proposed **LTRT+UNet** model with the prior method **SKF+UNet** on the **LOL-v1** dataset. This comparison highlights the effectiveness of the LTRT module as a plug-in enhancement for low-light image restoration. 

### SKF: Li X, Wang W, Hu X, et al. Selective kernel networks[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 510-519.

## Files

- `LOL_v1_skf_unet.py`  
  → Main training script for the SKF+UNet model on LOL-v1.

- `LOL_v1_best_skf_unet.pth`  
  → Pretrained model weights for reproducing quantitative results on LOL-v1 evaluation split.

- `LOL_v1_test_skf_unet.py`  
  → Evaluation and visualization script to reproduce reported PSNR / SSIM.

## How to Reproduce

1. **Prepare the LOL-v1 Dataset**

   LOL_v1 data is available at:

   Anonymous Link: https://pan.baidu.com/s/1F19nI5bG_BinyXJ0NtojGw?pwd=pfv2 Code: pfv2 

2. **Download Pretrained Model**

Place `LOL_v1_best_skf_unet.pth` in the root directory (or adjust path in code if necessary).

3. **Run Inference**

python LOL_v1_test_skf_unet.py

## Quantitative Results on LOL-v1 Dataset

| Model        | PSNR ↑ | SSIM ↑ |
|--------------|--------|--------|
| SKF+UNet | 20.52  | 0.86   |
| **LTRT+UNet** | **23.21**  | **0.88**   |

# LTRB Edge Detection (Unsupervised)

This project implements unsupervised edge detection using the Truncated Riesz Transform (LTRB) — a novel frequency-based technique designed for structural feature extraction without any labeled data. The method operates in the fractional Fourier domain, capturing edge orientation, amplitude, and phase details through adaptive Riesz kernels modulated by a truncated kernel function.

## Getting Started

### Install required dependencies:

- `pip install numpy opencv-python matplotlib scipy`

### Run the Code

- `python LTRB_edge_detect_unsupervision.py`

### You can replace the input with any grayscale image by changing the filename in the script:

- `img = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)`

### Output Example

![ex_3](https://github.com/user-attachments/assets/b57faca3-7d47-43d6-adfa-f119176a8e17)
