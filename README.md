# LTRT
**LTRT: Learnable Truncated Riesz Transform for Low-Light Enhancement and Beyond**

Official PyTorch implementation of our AAAI 2026 submission.
LTRT is a novel frequency-aware plugin module designed to improve segmentation accuracy by capturing edge and directional information through a truncated Riesz transform. It can be seamlessly inserted into existing segmentation models such as UNet, ResUNet, and ResUNet++.

**Highlights：**

1、Plug-and-Play Design: Integrates easily into mainstream segmentation backbones.

2、Frequency-Aware Edge Enhancement: Models directional and structural cues with low overhead.

![main_light](https://github.com/user-attachments/assets/c9ebb85b-653e-4af3-a410-ee280a7e5fa9)

**Inference & Reproduction Steps on LOL-v1 Dataset**

To reproduce the experimental results of LTRT + UNet on the LOL-v1 dataset, follow the steps below.

1. **Directory Structure**

Ensure the following files are present in the root directory:

── LOL_v1_best_ltrb_unet.pth                  # Pretrained model on LOL-v1

── LOL_v1_LTRB_unet_mid.py                    # Main code for inference on LOL-v1

── LOL_v1_test_LTRB_unet.py                   # Script to evaluate performance (PSNR, SSIM)

── datasets/LOL_v1/                           # LOL-v1 dataset directory. Dataset tip: You can download LOL-v1 from

2. **Run Inference**

Use the following command to run the inference on the LOL-v1 test set using LTRT-UNet and the pretrained weights:

python LOL_v1_LTRB_unet_mid.py \

  --weights LOL_v1_best_ltrb_unet.pth \
  
  --data_dir ./datasets/LOL_v1/Test/Low \
  
  --save_dir ./results/

3. **Evaluate Results**

To calculate PSNR and SSIM between the enhanced images and the ground truth:

python LOL_v1_test_LTRB_unet.py \

  --pred_dir ./results/ \
  
  --gt_dir ./datasets/LOL_v1/Test/Normal/



