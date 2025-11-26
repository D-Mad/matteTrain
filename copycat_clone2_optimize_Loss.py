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
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp  # Thư viện Backbone mạnh mẽ

# ==========================================
# PHẦN 1: DATASET (Xử lý Keyframe thông minh)
# ==========================================
class CopyCatDataset(Dataset):
    def __init__(self, img_dir, matte_dir, crop_size=512, epoch_len=1000, is_train=True):
        self.img_dir = img_dir
        self.matte_dir = matte_dir
        self.crop_size = crop_size
        self.epoch_len = epoch_len
        self.is_train = is_train

        # Chỉ lấy những file tồn tại trong cả thư mục ảnh và thư mục matte (Keyframes)
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
        
        self.pairs = []
        for f_name in all_imgs:
            matte_path = os.path.join(matte_dir, f_name)
            # Nếu file này có matte tương ứng -> Đây là keyframe
            if os.path.exists(matte_path):
                self.pairs.append(f_name)
        
        if len(self.pairs) == 0:
            raise ValueError(f"Không tìm thấy cặp ảnh/matte nào trùng tên trong {img_dir} và {matte_dir}")
        
        print(f"-> Found {len(self.pairs)} keyframes for training.")

    def __len__(self):
        # Trick: Dataset dài vô tận ảo để DataLoader chạy liên tục trong 1 epoch
        return self.epoch_len

    def __getitem__(self, idx):
        # Random chọn 1 keyframe từ danh sách
        f_name = random.choice(self.pairs)
        
        img_path = os.path.join(self.img_dir, f_name)
        matte_path = os.path.join(self.matte_dir, f_name)
        
        image = Image.open(img_path).convert("RGB")
        matte = Image.open(matte_path).convert("L")
        
        # --- Augmentation Logic ---
        # 1. Random Crop (Quan trọng cho ảnh 4K)
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        matte = TF.crop(matte, i, j, h, w)
        
        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            matte = TF.hflip(matte)
        
        # 3. Color Jitter (Giúp model không bị phụ thuộc vào ánh sáng cụ thể -> Giảm flicker)
        if self.is_train:
            # Random chỉnh độ sáng, tương phản nhẹ để model học đặc điểm vật thể thay vì màu sắc
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            image = jitter(image)
        
        return TF.to_tensor(image), TF.to_tensor(matte)

# ==========================================
# PHẦN 2: INFERENCE ENGINE (Tiled & Overlap)
# ==========================================
def predict_tiled(model, image_path, tile_size=768, overlap=0.25, device='cuda'):
    """
    Chiến thuật: Cắt ảnh to thành nhiều mảnh nhỏ (tile), predict từng mảnh,
    sau đó ghép lại có trọng số (Gaussian weights) để xóa lằn ranh.
    """
    model.eval()
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    
    full_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    # Bước nhảy (stride) nhỏ hơn tile_size để tạo vùng overlap
    stride = int(tile_size * (1 - overlap))
    
    # Tensor chứa kết quả và tensor đếm số lần cộng dồn
    output_map = torch.zeros((1, 1, H, W), device=device, dtype=torch.float16) # float16 tiết kiệm VRAM
    count_map = torch.zeros((1, 1, H, W), device=device, dtype=torch.float16)
    
    # Tạo mask trọng số (giữa tile trọng số cao, rìa tile trọng số thấp) để blend mượt
    # Đơn giản hóa: dùng mask bằng 1 (vẫn ổn với overlap 0.25) hoặc Gaussian
    base_weight = torch.ones((1, 1, tile_size, tile_size), device=device, dtype=torch.float16)

    with torch.no_grad():
        with autocast(): # Inference cũng dùng Mixed Precision
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    y1 = y
                    x1 = x
                    y2 = min(y + tile_size, H)
                    x2 = min(x + tile_size, W)
                    
                    # Điều chỉnh nếu tile ở sát mép cuối (lùi lại để lấy đủ kích thước)
                    if y2 - y1 < tile_size and H >= tile_size: y1 = H - tile_size
                    if x2 - x1 < tile_size and W >= tile_size: x1 = W - tile_size
                    y2 = y1 + tile_size
                    x2 = x1 + tile_size
                    
                    # Cắt Tile
                    tile = full_tensor[:, :, y1:y2, x1:x2]
                    
                    # Predict
                    pred_tile = model(tile)
                    pred_tile = torch.sigmoid(pred_tile) # Đưa về 0-1
                    
                    # Cộng dồn vào map tổng
                    output_map[:, :, y1:y2, x1:x2] += pred_tile * base_weight
                    count_map[:, :, y1:y2, x1:x2] += base_weight

    # Chia trung bình
    final_matte = (output_map / (count_map + 1e-8)).squeeze().float().cpu().numpy()
    return Image.fromarray((final_matte * 255).astype(np.uint8), mode='L')

def generate_preview(model, img_dir, save_dir, epoch, device, tile_size):
    """Tạo 1 ảnh preview để user biết model học đến đâu"""
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
    if not files: return
    
    # Lấy ảnh đầu tiên hoặc ngẫu nhiên để test
    test_img_path = os.path.join(img_dir, files[0]) # Luôn lấy file đầu để dễ so sánh
    
    preview_img = predict_tiled(model, test_img_path, tile_size=tile_size, overlap=0.25, device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"preview_ep{epoch:03d}.jpg")
    preview_img.save(save_path)

# ==========================================
# PHẦN 3: TRAINING & MAIN
# ==========================================
def train_mode(args):
    print(f"\n=== TRAINING: ResNet34 Backbone | Mixed Precision ===")
    print(f"Images: {args.images}")
    print(f"Mattes (Keyframes): {args.mattes}")
    
    # Setup thư mục
    base_dir = os.path.dirname(args.model_save) if os.path.dirname(args.model_save) else "."
    preview_dir = os.path.join(base_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)

    # 1. Model: Dùng SMP Unet với ResNet34
    # encoder_weights='imagenet': Giúp model đã biết nhìn ảnh, học cực nhanh
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
        activation=None # Để None vì ta sẽ dùng sigmoid + loss rời
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    if args.resume and os.path.exists(args.model_save):
        print(f"Resuming from {args.model_save}...")
        model.load_state_dict(torch.load(args.model_save))

    # 2. Dataset & DataLoader
    dataset = CopyCatDataset(args.images, args.mattes, crop_size=args.crop_size, epoch_len=args.steps_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Loss Combo: L1 (Pixel chính xác) + Dice (Hình khối chính xác)
    l1_loss = nn.L1Loss()
    dice_loss = smp.losses.DiceLoss(mode='binary')
    
    # 4. Mixed Precision Scaler
    scaler = GradScaler()

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch}/{args.epochs}")
        
        for imgs, mattes in pbar:
            imgs, mattes = imgs.to(device), mattes.to(device)
            
            optimizer.zero_grad()
            
            # Autocast: Chạy tính toán dưới dạng float16
            with autocast():
                preds = model(imgs)
                preds_sigmoid = torch.sigmoid(preds)
                
                # Tổng hợp loss: 50% pixel chính xác + 50% shape chính xác
                loss = l1_loss(preds_sigmoid, mattes) * 0.5 + dice_loss(preds_sigmoid, mattes) * 0.5

            # Scale loss và backprop (tránh underflow số học)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Save & Preview
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), args.model_save)
            
        if epoch % args.preview_freq == 0:
            generate_preview(model, args.images, preview_dir, epoch, device, tile_size=args.tile_size)
            
    print(f"Training Done! Model saved to: {args.model_save}")

def inference_mode(args):
    print(f"\n=== INFERENCE MODE ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)
    
    # Load đúng cấu trúc model lúc train
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    model.to(device)
    
    if not os.path.exists(args.model_load):
        print("ERROR: Model file not found!")
        return
    
    model.load_state_dict(torch.load(args.model_load, map_location=device))
    print(f"Loaded model: {args.model_load}")
    
    files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    for f in tqdm(files, desc="Rendering"):
        in_path = os.path.join(args.input, f)
        
        # Chạy Tiled Inference
        res = predict_tiled(model, in_path, tile_size=args.tile_size, overlap=0.25, device=device)
        
        # Save
        out_name = os.path.splitext(f)[0] + ".png"
        res.save(os.path.join(args.output, out_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- TRAIN ---
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--images', required=True, help='Folder chứa toàn bộ sequence ảnh gốc')
    p_train.add_argument('--mattes', required=True, help='Folder chứa các keyframe mattes (4-5 hình)')
    p_train.add_argument('--model_save', default='copycat_model.pth')
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--batch', type=int, default=8, help='Tăng lên 16 nếu VRAM > 12GB')
    p_train.add_argument('--crop_size', type=int, default=512, help='Kích thước crop khi train (quan trọng)')
    p_train.add_argument('--steps_per_epoch', type=int, default=200)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--resume', action='store_true')
    p_train.add_argument('--save_freq', type=int, default=10)
    p_train.add_argument('--preview_freq', type=int, default=5)
    p_train.add_argument('--tile_size', type=int, default=1024, help='Kích thước tile khi preview')

    # --- INFERENCE ---
    p_infer = subparsers.add_parser('inference')
    p_infer.add_argument('--input', required=True, help='Folder sequence ảnh gốc')
    p_infer.add_argument('--output', required=True)
    p_infer.add_argument('--model_load', required=True)
    p_infer.add_argument('--tile_size', type=int, default=1024, help='Nên để to (1024 hoặc 1536) nếu VRAM 24GB')

    args = parser.parse_args()

    # Tối ưu hóa CUDNN cho tốc độ
    torch.backends.cudnn.benchmark = True

    if args.command == 'train':
        train_mode(args)
    elif args.command == 'inference':
        inference_mode(args)
