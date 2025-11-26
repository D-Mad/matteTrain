"""
Temporal Consistency (Tính nhất quán theo thời gian).

Cơ chế hoạt động mới:
Input (4 Channels): RGB (3 kênh) + Previous Mask (1 kênh).

Khi Training (Giả lập): Vì chúng ta train trên từng ảnh rời rạc (keyframe), ta không có "kết quả frame trước" thực tế.

Giải pháp: Ta lấy chính cái Mask đích (Ground Truth), sau đó làm méo (perturb) nó đi (dịch chuyển nhẹ, làm mờ, thêm nhiễu) để giả làm "kết quả không hoàn hảo của frame trước".

Mục đích: Dạy model sửa lỗi từ mask cũ dựa trên hình ảnh RGB mới.
"""


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
from torch.cuda.amp import autocast, GradScaler 

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

# ==========================================
# HELPER: PERTURB MASK (Giả lập Mask frame trước)
# ==========================================
def perturb_mask(mask_tensor):
    """
    Làm biến dạng mask thật để giả lập 'previous mask' trong lúc train.
    Gồm: Dịch chuyển nhẹ, Scale nhẹ, hoặc xóa bớt chi tiết.
    """
    # mask_tensor: (1, H, W)
    if random.random() > 0.5:
        # Cách 1: Affine transform (Dịch chuyển/Xoay nhẹ)
        angle = random.uniform(-3, 3)
        translate = (random.randint(-5, 5), random.randint(-5, 5))
        scale = random.uniform(0.95, 1.05)
        mask_tensor = TF.affine(mask_tensor, angle=angle, translate=translate, scale=scale, shear=0)
    
    # Cách 2: Random Erasing (Mất một mảng nhỏ mask)
    if random.random() > 0.7:
        i, j, h, w = T.RandomCrop.get_params(mask_tensor, output_size=(mask_tensor.shape[1]//4, mask_tensor.shape[2]//4))
        mask_tensor[:, i:i+h, j:j+w] = 0 

    return mask_tensor

# ==========================================
# PHẦN 1: DATASET (4 CHANNELS)
# ==========================================
class CopyCatDataset(Dataset):
    def __init__(self, img_dir, matte_dir, crop_size=512, epoch_len=1000, is_train=True):
        self.img_dir = img_dir
        self.matte_dir = matte_dir
        self.crop_size = crop_size
        self.epoch_len = epoch_len
        self.is_train = is_train

        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
        
        self.pairs = []
        for f_name in all_imgs:
            matte_path = os.path.join(matte_dir, f_name)
            if os.path.exists(matte_path):
                self.pairs.append(f_name)
        
        if len(self.pairs) == 0:
            raise ValueError(f"Không tìm thấy keyframe nào trong {img_dir} và {matte_dir}")
        
        print(f"-> Found {len(self.pairs)} keyframes.")

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        f_name = random.choice(self.pairs)
        img_path = os.path.join(self.img_dir, f_name)
        matte_path = os.path.join(self.matte_dir, f_name)
        
        image = Image.open(img_path).convert("RGB")
        matte = Image.open(matte_path).convert("L")
        
        # 1. Random Crop đồng bộ
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        matte = TF.crop(matte, i, j, h, w)
        
        # 2. Horizontal Flip đồng bộ
        if random.random() > 0.5:
            image = TF.hflip(image)
            matte = TF.hflip(matte)
        
        # 3. Color Jitter (Chỉ áp dụng lên ảnh RGB)
        if self.is_train:
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            image = jitter(image)
        
        # Chuyển về Tensor
        img_t = TF.to_tensor(image)       # (3, H, W)
        matte_t = TF.to_tensor(matte)     # (1, H, W) -> Đây là Ground Truth (Target)

        # --- TẠO INPUT CHANNEL THỨ 4 (GUIDE MASK) ---
        if self.is_train:
            # Lúc train: Lấy matte thật làm méo đi để giả làm "kết quả cũ"
            guide_t = perturb_mask(matte_t.clone())
            
            # Đôi khi (20%) cho guide đen xì để model học cách tự xử lý khi mask cũ bị mất hoặc frame đầu
            if random.random() < 0.2:
                guide_t = torch.zeros_like(guide_t)
        else:
            # Validation giả: Dùng chính matte thật (lý tưởng)
            guide_t = matte_t.clone()

        # Nối RGB (3) + Guide (1) thành Input (4)
        input_t = torch.cat([img_t, guide_t], dim=0) # (4, H, W)
        
        return input_t, matte_t

# ==========================================
# PHẦN 2: INFERENCE ENGINE
# ==========================================
def get_gaussian_weight(size, device):
    x = torch.arange(size, device=device).float()
    center = size // 2
    sigma = size * 0.25
    gaussian_1d = torch.exp(- (x - center)**2 / (2 * sigma**2))
    weight = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    return weight.unsqueeze(0).unsqueeze(0)

def predict_tiled_4channel(model, img_path, prev_matte_tensor, tile_size=1024, overlap=0.25, device='cuda'):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    
    rgb_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    if prev_matte_tensor is None:
        guide_tensor = torch.zeros((1, 1, H, W), device=device)
    else:
        if prev_matte_tensor.shape[-2:] != (H, W):
            guide_tensor = TF.resize(prev_matte_tensor, (H, W))
        else:
            guide_tensor = prev_matte_tensor

    full_input = torch.cat([rgb_tensor, guide_tensor], dim=1)

    stride = int(tile_size * (1 - overlap))
    output_map = torch.zeros((1, 1, H, W), device=device, dtype=torch.float16)
    count_map = torch.zeros((1, 1, H, W), device=device, dtype=torch.float16)
    base_weight = get_gaussian_weight(tile_size, device).half()

    with torch.no_grad():
        with autocast(): 
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    y1 = y
                    x1 = x
                    y2 = min(y + tile_size, H)
                    x2 = min(x + tile_size, W)
                    
                    if y2 - y1 < tile_size and H >= tile_size: y1 = H - tile_size
                    if x2 - x1 < tile_size and W >= tile_size: x1 = W - tile_size
                    y2 = y1 + tile_size
                    x2 = x1 + tile_size
                    
                    tile = full_input[:, :, y1:y2, x1:x2]
                    pred_tile = model(tile)
                    pred_tile = torch.sigmoid(pred_tile)
                    
                    output_map[:, :, y1:y2, x1:x2] += pred_tile * base_weight
                    count_map[:, :, y1:y2, x1:x2] += base_weight

    final_matte = output_map / (count_map + 1e-8)
    return final_matte.float()

def generate_preview(model, img_dir, save_dir, epoch, device, tile_size):
    """Tạo ảnh preview. Lưu ý: Dùng mask đen làm guide để test khả năng 'cold start'"""
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
    if not files: return
    
    # Lấy ảnh đầu tiên để test
    test_img_path = os.path.join(img_dir, files[0]) 
    
    # Truyền prev_matte_tensor = None để giả lập frame đầu tiên (Guide đen)
    preview_tensor = predict_tiled_4channel(model, test_img_path, None, tile_size=tile_size, device=device)
    
    # Save
    preview_np = (preview_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    preview_img = Image.fromarray(preview_np, mode='L')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"preview_ep{epoch:03d}.jpg")
    preview_img.save(save_path)

# ==========================================
# PHẦN 3: TRAINING & MAIN
# ==========================================
def train_mode(args):
    print(f"\n=== TRAINING (4-Channel Temporal Input) ===")
    
    base_dir = os.path.dirname(args.model_save) if os.path.dirname(args.model_save) else "."
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    preview_dir = os.path.join(base_dir, "previews")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=4,   # 4 Kênh: RGB + Previous Mask
        classes=1,                      
        activation=None 
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    if args.resume and os.path.exists(args.model_save):
        print(f"Resuming from {args.model_save}...")
        model.load_state_dict(torch.load(args.model_save))

    dataset = CopyCatDataset(args.images, args.mattes, crop_size=args.crop_size, epoch_len=args.steps_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    l1_loss = nn.L1Loss()
    dice_loss = smp.losses.DiceLoss(mode='binary')
    scaler = GradScaler()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch}/{args.epochs}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(inputs)
                preds_sigmoid = torch.sigmoid(preds)
                loss = l1_loss(preds_sigmoid, targets) * 0.5 + dice_loss(preds_sigmoid, targets) * 0.5

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), args.model_save)
            # Lưu thêm checkpoint đánh số
            ckpt_name = f"ckpt_ep{epoch:03d}.pth"
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, ckpt_name))
        
        # --- PREVIEW LOGIC (Đã thêm lại) ---
        if epoch % args.preview_freq == 0:
            generate_preview(model, args.images, preview_dir, epoch, device, tile_size=args.tile_size)
            
    print(f"Training Done! Model saved to: {args.model_save}")

def inference_mode(args):
    print(f"\n=== INFERENCE (Sequential Propagation) ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)
    
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1, activation=None)
    model.to(device)
    model.load_state_dict(torch.load(args.model_load, map_location=device))
    
    files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    
    prev_matte = None 
    
    for f in tqdm(files, desc="Rendering Sequence"):
        in_path = os.path.join(args.input, f)
        matte_tensor = predict_tiled_4channel(model, in_path, prev_matte, tile_size=args.tile_size, device=device)
        prev_matte = matte_tensor
        
        out_np = (matte_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        out_img = Image.fromarray(out_np, mode='L')
        out_img.save(os.path.join(args.output, os.path.splitext(f)[0] + ".png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- TRAIN ---
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--images', required=True)
    p_train.add_argument('--mattes', required=True)
    p_train.add_argument('--model_save', default='copycat_4ch.pth')
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--batch', type=int, default=8)
    p_train.add_argument('--crop_size', type=int, default=512)
    p_train.add_argument('--steps_per_epoch', type=int, default=200)
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--resume', action='store_true')
    p_train.add_argument('--save_freq', type=int, default=10)
    # Đã thêm lại tham số này:
    p_train.add_argument('--preview_freq', type=int, default=5)
    p_train.add_argument('--tile_size', type=int, default=1024)

    # --- INFERENCE ---
    p_infer = subparsers.add_parser('inference')
    p_infer.add_argument('--input', required=True)
    p_infer.add_argument('--output', required=True)
    p_infer.add_argument('--model_load', required=True)
    p_infer.add_argument('--tile_size', type=int, default=1024)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    if args.command == 'train':
        train_mode(args)
    elif args.command == 'inference':
        inference_mode(args)