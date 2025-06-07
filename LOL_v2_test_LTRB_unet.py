import random
import numpy as np
import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
from torchvision.models import vgg16
from torch.utils.data import DataLoader
import pytest

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LOLV2RealDataset(torch.utils.data.Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_paths = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg'))])
        self.high_paths = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg'))])
        self.transform = transform
    def __len__(self):
        return len(self.low_paths)
    def __getitem__(self, idx):
        low = Image.fromarray(cv2.cvtColor(cv2.imread(self.low_paths[idx]), cv2.COLOR_BGR2RGB))
        high = Image.fromarray(cv2.cvtColor(cv2.imread(self.high_paths[idx]), cv2.COLOR_BGR2RGB))
        if self.transform:
            low, high = self.transform(low), self.transform(high)
        else:
            low, high = ToTensor()(low), ToTensor()(high)
        return low, high

def load_model(model_class, model_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def inference_on_test(model, test_loader, device, save_dir="output_results"):
    os.makedirs(save_dir, exist_ok=True)
    psnr_total, ssim_total, count = 0.0, 0.0, 0
    with torch.no_grad():
        for idx, (low, high) in enumerate(test_loader):
            low, high = low.to(device), high.to(device)
            output = torch.clamp(model(low), 0, 1)
            output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            high_np = high.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            psnr_total += psnr(high_np, output_np, data_range=1.0)
            ssim_total += ssim(high_np, output_np, channel_axis=-1, data_range=1.0)
            count += 1
            output_img = (output_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"result_{idx:04d}.png"), cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count
    return avg_psnr, avg_ssim

@pytest.mark.inference
def test_inference():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from LOL_v2_LTRB_unet_mid import LTRB_UNet
    model = load_model(lambda: LTRB_UNet(in_ch=3, out_ch=3, height=128, width=128), "LOL_v2_best_ltrb_unet.pth", device)
    transform = Compose([Resize((128, 128)), ToTensor()])
    test_dataset = LOLV2RealDataset("D:/iccv2025/datasets/LOL-v2/Real_captured/Test/Low", "D:/iccv2025/datasets/LOL-v2/Real_captured/Test/Normal", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    avg_psnr, avg_ssim = inference_on_test(model, test_loader, device)
    print(f"Test PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    assert avg_psnr > 15, "PSNR too low"
    assert avg_ssim > 0.5, "SSIM too low"
