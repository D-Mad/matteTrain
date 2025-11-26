import os
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import torchvision modules chuẩn xác
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ==========================================
# PHẦN 1: MODEL U-NET
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        if x.shape != x3.shape: x = TF.resize(x, size=x3.shape[2:])
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        if x.shape != x2.shape: x = TF.resize(x, size=x2.shape[2:])
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        if x.shape != x1.shape: x = TF.resize(x, size=x1.shape[2:])
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        return torch.sigmoid(self.outc(x))

# ==========================================
# PHẦN 2: DATASET
# ==========================================
class CopyCatDataset(Dataset):
    def __init__(self, img_dir, matte_dir, crop_size=256, epoch_len=1000):
        self.img_dir = img_dir
        self.matte_dir = matte_dir
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.img_ids = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
        self.matte_ids = sorted([f for f in os.listdir(matte_dir) if f.lower().endswith(valid_ext)])
        self.crop_size = crop_size
        self.epoch_len = epoch_len 
        
        if len(self.img_ids) != len(self.matte_ids):
            print(f"WARNING: Số lượng ảnh ({len(self.img_ids)}) và matte ({len(self.matte_ids)}) không khớp!")

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        real_idx = random.randint(0, len(self.img_ids) - 1)
        img_path = os.path.join(self.img_dir, self.img_ids[real_idx])
        matte_path = os.path.join(self.matte_dir, self.matte_ids[real_idx])
        
        image = Image.open(img_path).convert("RGB")
        matte = Image.open(matte_path).convert("L")
        
        # Random Crop & Flip
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        matte = TF.crop(matte, i, j, h, w)
        
        if random.random() > 0.5:
            image = TF.hflip(image)
            matte = TF.hflip(matte)
            
        # Color Jitter
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        image = jitter(image)
        
        return TF.to_tensor(image), TF.to_tensor(matte)

# ==========================================
# PHẦN 3: TILED INFERENCE & PREVIEW
# ==========================================
def predict_tiled(model, image_path, tile_size=512, overlap=0.25, device='cuda'):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    
    full_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    stride = int(tile_size * (1 - overlap))
    
    output_map = torch.zeros((1, 1, H, W), device=device)
    count_map = torch.zeros((1, 1, H, W), device=device)
    
    # Tạo weight mask gốc
    base_weight_mask = torch.ones((1, 1, tile_size, tile_size), device=device)

    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1 = y
                x1 = x
                # Logic cũ: y2 = min(y + tile_size, H) -> gây lỗi nếu H < tile_size
                
                # Logic mới: Cố gắng lấy đủ tile_size, nhưng nếu ảnh nhỏ hơn thì chịu
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)
                
                # Backtracking: Nếu crop bị hụt ở cuối, lùi lại để lấy đủ tile (chỉ khi ảnh lớn hơn tile)
                if y2 - y1 < tile_size and H >= tile_size: y1 = H - tile_size
                if x2 - x1 < tile_size and W >= tile_size: x1 = W - tile_size
                
                # Tính lại y2, x2 sau khi lùi (để chắc chắn)
                y2 = min(y1 + tile_size, H)
                x2 = min(x1 + tile_size, W)

                # Crop thực tế
                tile = full_tensor[:, :, y1:y2, x1:x2]
                
                # Inference
                pred_tile = model(tile)
                
                # --- PHẦN SỬA LỖI QUAN TRỌNG ---
                # Lấy kích thước thực tế của tile vừa chạy ra (có thể là 480 thay vì 512)
                h_actual, w_actual = pred_tile.shape[2], pred_tile.shape[3]
                
                # Cắt mask cho khớp với kích thước thực tế
                current_mask = base_weight_mask[:, :, :h_actual, :w_actual]
                
                # Cộng dồn (Bây giờ kích thước đã khớp 100%)
                output_map[:, :, y1:y1+h_actual, x1:x1+w_actual] += pred_tile * current_mask
                count_map[:, :, y1:y1+h_actual, x1:x1+w_actual] += current_mask

    final_matte = (output_map / (count_map + 1e-8)).squeeze().cpu().numpy()
    return Image.fromarray((final_matte * 255).astype(np.uint8), mode='L')

def generate_preview(model, img_dir, save_dir, epoch, device):
    """Hàm tạo ảnh preview nhanh để kiểm tra chất lượng"""
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
    if not files: return
    
    # Lấy ảnh đầu tiên làm mẫu test
    test_img_path = os.path.join(img_dir, files[0])
    
    # Chạy inference (dùng tile nhỏ 512 để nhanh)
    preview_img = predict_tiled(model, test_img_path, tile_size=512, device=device)
    
    # Lưu ảnh
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"preview_ep{epoch:03d}.png")
    preview_img.save(save_path)

# ==========================================
# PHẦN 4: LOGIC TRAIN & INFERENCE
# ==========================================
def train_mode(args):
    print(f"\n=== TRAINING MODE ===")
    print(f"Images: {args.images} | Mattes: {args.mattes}")
    
    # Tạo thư mục lưu checkpoints và previews
    checkpoint_dir = os.path.join(os.path.dirname(args.model_save), "checkpoints")
    preview_dir = os.path.join(os.path.dirname(args.model_save), "previews")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    dataset = CopyCatDataset(args.images, args.mattes, crop_size=args.crop_size, epoch_len=args.steps_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    if args.resume and os.path.exists(args.model_save):
        print(f"Resuming from: {args.model_save}")
        model.load_state_dict(torch.load(args.model_save))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for imgs, mattes in pbar:
            imgs, mattes = imgs.to(device), mattes.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), mattes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 1. LƯU CHECKPOINT THEO CH Chu kỳ
        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_ep{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            # Luôn update file model chính
            torch.save(model.state_dict(), args.model_save)
            
        # 2. TẠO PREVIEW THEO CHU KỲ
        if epoch % args.preview_freq == 0:
            generate_preview(model, args.images, preview_dir, epoch, device)
            
    print(f"Done! Final model: {args.model_save}")
    print(f"Checkpoints in: {checkpoint_dir}")
    print(f"Previews in: {preview_dir}")

def inference_mode(args):
    print(f"\n=== INFERENCE MODE ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)
    
    model = UNet(n_channels=3, n_classes=1).to(device)
    if not os.path.exists(args.model_load):
        print(f"ERROR: File model {args.model_load} không tồn tại.")
        return
    
    model.load_state_dict(torch.load(args.model_load, map_location=device))
    
    files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for f in tqdm(files, desc="Processing"):
        res = predict_tiled(model, os.path.join(args.input, f), tile_size=args.tile_size, device=device)
        res.save(os.path.join(args.output, os.path.splitext(f)[0] + ".png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # TRAIN ARGS
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--images', required=True, help='Folder ảnh training')
    p_train.add_argument('--mattes', required=True, help='Folder matte training')
    p_train.add_argument('--model_save', default='my_model.pth')
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch', type=int, default=4)
    p_train.add_argument('--crop_size', type=int, default=384)
    p_train.add_argument('--steps_per_epoch', type=int, default=500)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--resume', action='store_true')
    # Hai tham số mới
    p_train.add_argument('--save_freq', type=int, default=10, help='Lưu model mỗi X epoch')
    p_train.add_argument('--preview_freq', type=int, default=5, help='Xuất ảnh preview mỗi X epoch')

    # INFERENCE ARGS
    p_infer = subparsers.add_parser('inference')
    p_infer.add_argument('--input', required=True)
    p_infer.add_argument('--output', required=True)
    p_infer.add_argument('--model_load', required=True)
    p_infer.add_argument('--tile_size', type=int, default=768)

    args = parser.parse_args()

    if args.command == 'train':
        train_mode(args)
    elif args.command == 'inference':
        inference_mode(args)
