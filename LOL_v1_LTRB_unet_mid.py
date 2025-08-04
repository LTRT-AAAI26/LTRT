# ------------------------------
# Imports
# ------------------------------
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
from torchvision.models import vgg16

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ------------------------------
# Dataset
# ------------------------------
class LOLV1RealDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_paths = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg'))])
        self.high_paths = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir) if f.endswith(('.png', '.jpg'))])
        self.transform = transform

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = cv2.cvtColor(cv2.imread(self.low_paths[idx]), cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(cv2.imread(self.high_paths[idx]), cv2.COLOR_BGR2RGB)
        low, high = Image.fromarray(low), Image.fromarray(high)
        if self.transform:
            low, high = self.transform(low), self.transform(high)
        return low, high

# ------------------------------
# LTRT (Vectorized)
# ------------------------------
def generate_riesz_kernels(H, W, device, p=(1,1)):
    Y, X = torch.meshgrid(torch.arange(H, device=device)+1, torch.arange(W, device=device)+1, indexing='ij')
    a1, a2 = 0.5 * torch.tensor(torch.pi, device=device) * p[0], 0.5 * torch.tensor(torch.pi, device=device) * p[1]
    sin_a1, sin_a2 = torch.sin(a1), torch.sin(a2)
    denom = torch.sqrt((X / sin_a1)**2 + (Y / sin_a2)**2) + 1e-6
    dx = -1j * (X / sin_a1) / denom
    dy = -1j * (Y / sin_a2) / denom
    return dx, dy

def frft2d(matrix, angles):
    matrix = matrix.to(torch.complex64)
    temp = matrix.transpose(-2, -1).contiguous()
    for a in angles:
        N = temp.shape[-2]
        a %= 4
        shft = torch.remainder(torch.arange(N) + N//2, N).to(matrix.device)
        sN = torch.sqrt(torch.tensor(N, dtype=matrix.dtype, device=matrix.device))
        if a == 1:
            temp[shft, :] = torch.fft.fft(temp[shft, :], dim=-2) / sN
        elif a == 2:
            temp = torch.flip(temp, dims=[-2])
        elif a == 3:
            temp[shft, :] = torch.fft.ifft(temp[shft, :], dim=-2) * sN
        temp = temp.transpose(-2, -1).contiguous()
    return temp

class RieszFeatureEnhancer(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.kernel_map = nn.Parameter(torch.ones(height, width))
        self.theta_conv = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.Sigmoid())
        self.phase_fusion = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.BatchNorm2d(1), nn.ReLU(inplace=True))
        self.amplitude_boost = nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU(inplace=True))
        
        self.final_fusion = nn.Sequential(nn.Conv2d(6, 1, 1), nn.BatchNorm2d(1), nn.ReLU(inplace=True))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(1, 1, kernel_size=3, padding=3, dilation=3)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B*C, H, W)
        trker = F.softplus(self.kernel_map.to(x.device))
        trker = torch.clamp(trker, 1e-3, 10.0)
        dx, dy = generate_riesz_kernels(H, W, x.device)
        dx, dy = dx * trker, dy * trker
        fa = frft2d(x_flat, [1,1])
        R1 = torch.abs(frft2d(dx*fa, [-1,-1])).real + 1e-6
        R2 = torch.abs(frft2d(dy*fa, [-1,-1])).real + 1e-6
        A = torch.sqrt(F.relu(x_flat**2 + R1**2 + R2**2) + 1e-6).unsqueeze(1)
        P = torch.atan2(torch.sqrt(R1**2 + R2**2) + 1e-6, x_flat+1e-6).unsqueeze(1)
        theta = torch.atan2(R2, R1+1e-6).unsqueeze(1)
        G_theta = theta * self.theta_conv(theta)
        P_prime = self.phase_fusion(P)
        A_boost = A + self.amplitude_boost(A)
        multi_scale_feats = [conv(A_boost) for conv in self.multi_scale_conv]
        multi_scale = torch.cat(multi_scale_feats, dim=1)
        multi_scale = F.adaptive_avg_pool2d(multi_scale, (H, W))
        global_feat = self.global_avg_pool(A_boost)
        global_weight = self.global_fc(global_feat.view(B*C, -1)).view(B*C,1,1,1)
        multi_scale = multi_scale * global_weight
        fusion_input = torch.cat([A_boost, P_prime, G_theta, multi_scale], dim=1)
        fused = self.final_fusion(fusion_input)
        fused = torch.clamp(fused, 0.0, 1.0)
        return fused.view(B, C, H, W)

# ------------------------------
# UNet Core
# ------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class LTRB_UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, height=128, width=128):
        super().__init__()
        self.inc = ConvBlock(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.ltrb = RieszFeatureEnhancer(height//4, width//4)
        self.up1 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.ltrb(x3)  
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)


# ------------------------------
# Loss Function
# ------------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x, y):
        return F.l1_loss(self.vgg(self.normalize(x)), self.vgg(self.normalize(y)))

def freq_loss(pred, target):
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

def total_loss(pred, target, perceptual_loss):
    l1 = F.l1_loss(pred, target)
    ssim_l = 1 - ms_ssim(pred, target, data_range=1.0, win_size=7, size_average=True)
    percept = perceptual_loss(pred, target)
    return 0.4 * l1 + 0.3 * ssim_l + 0.3 * percept

def compute_psnr_ssim(pred, gt):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    gt_np = gt.detach().cpu().numpy().transpose(1, 2, 0)
    return psnr(gt_np, pred_np, data_range=1.0), ssim(gt_np, pred_np, channel_axis=-1, data_range=1.0)

# ------------------------------
# Evaluation & Training
# ------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    psnr_total, ssim_total, count = 0.0, 0.0, 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = torch.clamp(model(low), 0, 1)
            output_np = output.cpu().numpy().transpose(0, 2, 3, 1)
            high_np = high.cpu().numpy().transpose(0, 2, 3, 1)
            for i in range(output_np.shape[0]):
                psnr_total += psnr(high_np[i], output_np[i], data_range=1.0)
                ssim_total += ssim(high_np[i], output_np[i], channel_axis=-1, data_range=1.0)
                count += 1
    return psnr_total / count, ssim_total / count

# ------------------------------
# Training Loop
# ------------------------------
def train_ltrb_unet(model, train_loader, test_loader, device, epochs=300):
    perceptual_loss = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_psnr = 0.0
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (low, high) in enumerate(train_loader):
            low, high = low.to(device), high.to(device)
            output = torch.clamp(model(low), 0.0, 1.0)
            loss = total_loss(output, high, perceptual_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        psnr_val, ssim_val = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - PSNR: {psnr_val:.2f} - SSIM: {ssim_val:.4f}")

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "LOL_v1_best_ltrb_unet.pth")
            print(f"✅ Saved new best model @ Epoch {epoch+1}, PSNR: {best_psnr:.2f}")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    transform = Compose([Resize((128, 128)), ToTensor()])
    # train_dataset = LOLV2RealDataset(
    #     r"D:/iccv2025/datasets/LOL-v2/Real_captured/Train/Low",
    #     r"D:/iccv2025/datasets/LOL-v2/Real_captured/Train/Normal",
    #     transform=transform)
    # test_dataset = LOLV2RealDataset(
    #     r"D:/iccv2025/datasets/LOL-v2/Real_captured/Test/Low",
    #     r"D:/iccv2025/datasets/LOL-v2/Real_captured/Test/Normal",
    #     transform=transform)

    # LOL_v1
    train_dataset = LOLV1RealDataset(
        r"D:/AAAI2026/datasets/LOL_v1/train485/Low",
        r"D:/AAAI2026/datasets/LOL_v1/train485/high",
        transform=transform)
    test_dataset = LOLV1RealDataset(
        r"D:/AAAI2026/datasets/LOL_v1/eval15/Low",
        r"D:/AAAI2026/datasets/LOL_v1/eval15/high",
        transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LTRB_UNet(in_ch=3, out_ch=3, height=128, width=128).to(device)
    train_ltrb_unet(model, train_loader, test_loader, device)

