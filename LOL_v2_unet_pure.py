# ------------------------------
# Imports
# ------------------------------
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
from torchvision.models import vgg16
import copy

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
class LOLV2RealDataset(Dataset):
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
# Pure UNet (Baseline without Riesz)
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

class PureUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.inc = ConvBlock(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.up1 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
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

def total_loss(pred, target, perceptual_loss):
    l1 = F.l1_loss(pred, target)
    ssim_loss = 1 - ms_ssim(pred, target, data_range=1.0, win_size=7, size_average=True)
    percept = perceptual_loss(pred, target)
    return 0.4 * l1 + 0.3 * ssim_loss + 0.3 * percept

def compute_psnr_ssim(pred, gt):
    pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
    gt_np = gt.detach().cpu().numpy().transpose(1, 2, 0)
    return psnr(gt_np, pred_np, data_range=1.0), ssim(gt_np, pred_np, channel_axis=-1, data_range=1.0)

# ------------------------------
# Evaluation
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
# Training Loop (for Pure UNet)
# ------------------------------
def train_pure_unet(model, train_loader, test_loader, device, epochs=300):
    perceptual_loss = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_psnr = 0.0
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for low, high in train_loader:
            low, high = low.to(device), high.to(device)
            output = torch.clamp(model(low), 0.0, 1.0)
            loss = total_loss(output, high, perceptual_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        psnr_val, ssim_val = evaluate_model(model, test_loader, device)
        print(f"[PureUNet] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - PSNR: {psnr_val:.2f} - SSIM: {ssim_val:.4f}")

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, "best_pure_unet.pth")
            print(f"✅ Saved new best PureUNet model @ Epoch {epoch+1}, PSNR: {best_psnr:.2f}")

# ------------------------------
# Main Run
# ------------------------------
if __name__ == "__main__":
    transform = Compose([Resize((128, 128)), ToTensor()])
    train_dataset = LOLV2RealDataset(
        r"D:/iccv2025/datasets/LOL-v2/Real_captured/Train/Low",
        r"D:/iccv2025/datasets/LOL-v2/Real_captured/Train/Normal",
        transform=transform)
    test_dataset = LOLV2RealDataset(
        r"D:/iccv2025/datasets/LOL-v2/Real_captured/Test/Low",
        r"D:/iccv2025/datasets/LOL-v2/Real_captured/Test/Normal",
        transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n🔬 Training PureUNet (baseline)...")
    model_pure = PureUNet(in_ch=3, out_ch=3).to(device)
    train_pure_unet(model_pure, train_loader, test_loader, device, epochs=300)