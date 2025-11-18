import argparse
import os
import random
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Thêm import cho TensorBoard
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image

from res_unet_model import ResUNet # Thay thế MSRN bằng ResUNet
from dataset import AlphaMatteSequenceDataset
from losses import AlphaL1Loss, CompositionalLoss, LaplacianPyramidLoss # Bỏ GradientLoss, dùng LaplacianPyramidLoss


def parse_crop_sizes(crop_values):
    """
    Chuyển danh sách crop từ CLI thành list các tuple (h, w).
    - Truyền '256' sẽ hiểu là 256x256 (vuông)
    - Truyền '384x640' sẽ crop hình chữ nhật
    - Truyền 'full' => dùng toàn bộ khung hình (không crop).
    """
    parsed = []
    for value in crop_values:
        token = str(value).lower()
        if token in {'full', 'all', 'orig', 'original'}:
            parsed.append(None)
            continue

        if 'x' in token:
            try:
                h_str, w_str = token.split('x')
                parsed.append((int(h_str), int(w_str)))
            except ValueError:
                raise ValueError(f'Invalid crop specification: {value}')
        else:
            size = int(token)
            parsed.append((size, size))
    return parsed


def get_args():
    parser = argparse.ArgumentParser(description='Train the ResUNet model for Alpha Matting')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # --img-size is no longer needed for training, but kept for apply_model.py consistency
    # parser.add_argument('--img-size', type=int, default=512, help='Image size for training')
    parser.add_argument('--crop-sizes', type=str, nargs='+', default=['64', '128', '256', '512'], help="Danh sách crop (ví dụ: 256 384x640 full)")
    parser.add_argument('--data-path', type=str, default='data/train', help='Path to the training data directory')
    parser.add_argument('--bg-path', type=str, default=None, help='Path to the background images directory (optional)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10, help='Save a checkpoint every N epochs')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--visualize-dir', type=str, default='monitoring', help='Directory to save visualization images')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save TensorBoard logs')
    parser.add_argument('--viz-every', type=int, default=200, help='Số bước giữa các lần log crop lên TensorBoard (0 để tắt)')
    parser.add_argument('--viz-samples', type=int, default=3, help='Số crop hiển thị trên grid (tối đa mỗi lần log)')
    parser.add_argument('--viz-full-every', type=int, default=0, help='Tần suất (theo bước) chạy inference full frame để log (0 = tắt)')
    parser.add_argument('--viz-tile-size', type=int, default=512, help='Tile size dùng cho full-frame visualization')
    parser.add_argument('--viz-tile-overlap', type=int, default=64, help='Độ chồng tile cho visualization inference')
    parser.add_argument('--full-max-side', type=int, default=0, help='Giới hạn chiều dài lớn nhất khi dùng crop full (0 = không giới hạn)')
    parser.add_argument('--full-max-pixels', type=int, default=0, help='Giới hạn số pixel khi dùng crop full (0 = không giới hạn)')
    parser.add_argument('--background-mode', type=str, choices=['none', 'random_color', 'dataset'], default='random_color', help='Phương án sinh background cho Compositional Loss')
    
    # Thêm trọng số cho các hàm loss
    parser.add_argument('--weight-alpha', type=float, default=1.0, help='Weight for L1 Alpha Loss (w_alpha)')
    parser.add_argument('--weight-comp', type=float, default=1.0, help='Weight cho Compositional Loss (w_c)')
    parser.add_argument('--weight-lap', type=float, default=1.0, help='Weight for Laplacian Pyramid Loss (w_lap)')

    args = parser.parse_args()
    args.crop_sizes = parse_crop_sizes(args.crop_sizes)
    return args

def get_intelligent_crop_params(matte_pil, output_size, unknown_pixel_prob=0.75):
    w, h = matte_pil.size
    crop_h, crop_w = output_size
    
    if crop_h > h: crop_h = h
    if crop_w > w: crop_w = w

    is_random_crop = True
    if random.random() < unknown_pixel_prob:
        matte_np = np.array(matte_pil)
        unknown_pixels = np.where((matte_np > 0) & (matte_np < 255))
        
        if len(unknown_pixels[0]) > 0:
            is_random_crop = False
            idx = random.randint(0, len(unknown_pixels[0]) - 1)
            center_y, center_x = unknown_pixels[0][idx], unknown_pixels[1][idx]
            top = max(0, min(center_y - crop_h // 2, h - crop_h))
            left = max(0, min(center_x - crop_w // 2, w - crop_w))

    if is_random_crop:
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
    return top, left, crop_h, crop_w


def create_comparison_grid(batch, pred_alpha, num_samples=3):
    """Tạo grid 3x3 (input, groundtruth, output) cho nhiều crop."""
    max_samples = min(num_samples, batch['image_curr'].size(0))
    if max_samples == 0:
        raise ValueError('Batch rỗng, không thể visualize.')

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    red_bg = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)

    panels = []
    for idx in range(max_samples):
        input_img = batch['image_curr'][idx].cpu()
        gt_matte = batch['matte_gt'][idx].cpu()
        pred_matte = pred_alpha[idx].cpu()

        input_img_denorm = (input_img * std + mean).clamp(0, 1)
        gt_matte_colored = red_bg * (1 - gt_matte)
        pred_matte_colored = red_bg * (1 - pred_matte)

        panels.extend([input_img_denorm, gt_matte_colored, pred_matte_colored])

    grid = vutils.make_grid(panels, nrow=3, padding=2, normalize=False)
    return grid


def visualize_output(batch, pred_alpha, epoch, visualize_dir, num_samples=3):
    grid = create_comparison_grid(batch, pred_alpha, num_samples)
    vutils.save_image(grid, os.path.join(visualize_dir, f'epoch_{epoch+1:04d}.png'))


def custom_collate_fn(batch, crop_sizes, full_resize_cfg=None, background_mode='none'):
    # 1. Chọn một kích thước crop cho TOÀN BỘ batch này
    crop_spec = random.choice(crop_sizes)
    max_side = None
    max_pixels = None
    if full_resize_cfg:
        max_side = full_resize_cfg.get('max_side') or None
        max_pixels = full_resize_cfg.get('max_pixels') or None

    # 2. Định nghĩa các transform
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 3. Xử lý từng sample trong batch
    batch_input_9c = []
    batch_image_curr = []
    batch_matte_gt = []
    batch_background = []
    meta_info = []
    for item in batch:
        paths = item['paths']
        meta_info.append(paths.copy())

        # Tải ảnh
        img_prev = Image.open(paths['prev']).convert('RGB')
        img_curr = Image.open(paths['curr']).convert('RGB')
        img_next = Image.open(paths['next']).convert('RGB')
        matte_curr = Image.open(paths['matte']).convert('L')

        # Lấy tham số crop (h, w hoặc full frame)
        if crop_spec is None:
            i, j = 0, 0
            h, w = matte_curr.height, matte_curr.width
            resize_scale = 1.0
            if max_side:
                resize_scale = min(resize_scale, max_side / max(h, w))
            if max_pixels and h * w > 0:
                resize_scale = min(resize_scale, math.sqrt(max_pixels / (h * w)))
        else:
            i, j, h, w = get_intelligent_crop_params(matte_curr, crop_spec)
            resize_scale = 1.0

        new_w, new_h = w, h
        if resize_scale < 1.0:
            new_w = max(1, int(w * resize_scale))
            new_h = max(1, int(h * resize_scale))

        # Crop tất cả các ảnh theo cùng một tham số
        img_prev_cropped = TF.crop(img_prev, i, j, h, w)
        img_curr_cropped = TF.crop(img_curr, i, j, h, w)
        img_next_cropped = TF.crop(img_next, i, j, h, w)
        matte_curr_cropped = TF.crop(matte_curr, i, j, h, w)

        # Nếu đang dùng full frame và cần thu nhỏ để tránh OOM
        if resize_scale < 1.0:
            img_prev_cropped = img_prev_cropped.resize((new_w, new_h), Image.BILINEAR)
            img_curr_cropped = img_curr_cropped.resize((new_w, new_h), Image.BILINEAR)
            img_next_cropped = img_next_cropped.resize((new_w, new_h), Image.BILINEAR)
            matte_curr_cropped = matte_curr_cropped.resize((new_w, new_h), Image.NEAREST)
        # Chuyển đổi sang tensor
        t_prev = to_tensor(img_prev_cropped)
        t_curr = to_tensor(img_curr_cropped)
        t_next = to_tensor(img_next_cropped)
        t_matte = to_tensor(matte_curr_cropped)

        # Xếp chồng và thêm vào list
        # Chuẩn hóa ảnh trước khi xếp chồng
        input_tensor_9c = torch.cat([
            normalize(t_prev), 
            normalize(t_curr), 
            normalize(t_next)
        ], dim=0)
        
        batch_input_9c.append(input_tensor_9c)
        batch_image_curr.append(normalize(t_curr))
        batch_matte_gt.append(t_matte)

        if background_mode == 'dataset' and item.get('bg_path'):
            img_bg = Image.open(item['bg_path']).convert('RGB')
            img_bg_cropped = TF.crop(img_bg, i, j, h, w)
            if resize_scale < 1.0:
                img_bg_cropped = img_bg_cropped.resize((new_w, new_h), Image.BILINEAR)
            t_bg = to_tensor(img_bg_cropped)
        elif background_mode == 'random_color':
            t_bg = torch.rand(3, new_h, new_w)
        else:
            t_bg = torch.zeros(3, new_h, new_w)

        batch_background.append(normalize(t_bg))

    # 4. Stack tất cả các tensor lại thành một batch duy nhất
    return {
        'input_9c': torch.stack(batch_input_9c),
        'image_curr': torch.stack(batch_image_curr),
        'matte_gt': torch.stack(batch_matte_gt),
        'background': torch.stack(batch_background),
        'meta': meta_info
    }


def inference_sliding_window(model, input_tensor, tile_size, overlap, device):
    B, C, H, W = input_tensor.shape
    stride = tile_size - overlap
    output_matte = torch.zeros((B, 1, H, W), device=device)
    weight_map = torch.zeros((B, 1, H, W), device=device)

    blend_window = torch.bartlett_window(tile_size, periodic=False, device=device)
    blend_window_2d = torch.outer(blend_window, blend_window).unsqueeze(0).unsqueeze(0)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_start, y_end = y, min(y + tile_size, H)
            x_start, x_end = x, min(x + tile_size, W)

            tile = input_tensor[:, :, y_start:y_end, x_start:x_end]
            h_tile, w_tile = tile.shape[2:]
            pad_h = tile_size - h_tile
            pad_w = tile_size - w_tile
            if pad_h > 0 or pad_w > 0:
                if (pad_h > 0 and pad_h >= h_tile) or (pad_w > 0 and pad_w >= w_tile):
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='constant', value=0)
                else:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            pred_tile = model(tile)
            pred_tile = pred_tile[:, :, :h_tile, :w_tile]
            current_window = blend_window_2d[:, :, :h_tile, :w_tile]
            output_matte[:, :, y_start:y_end, x_start:x_end] += pred_tile * current_window
            weight_map[:, :, y_start:y_end, x_start:x_end] += current_window

    final_matte = output_matte / (weight_map + 1e-8)
    return final_matte.clamp(0, 1)


def train_model(args):
    # ---- 1. Setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Khởi tạo TensorBoard Writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f'resunet_lr{args.learning_rate}_bs{args.batch_size}'))
    print(f"TensorBoard logs will be saved to: {writer.log_dir}")

    # Sử dụng ResUNet thay vì MSRN
    model = ResUNet(
        n_channels_in=9, 
        n_classes_out=1
    ).to(device)
    
    if args.load:
        print(f'Loading model from {args.load}')
        model.load_state_dict(torch.load(args.load, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    scaler = GradScaler()

    # Loss functions - Triển khai theo hướng dẫn
    l1_loss_fn = AlphaL1Loss().to(device)
    comp_loss_fn = CompositionalLoss().to(device)
    lap_loss_fn = LaplacianPyramidLoss(device=device).to(device) # Sử dụng Laplacian Loss

    # Dataloaders
    train_dataset = AlphaMatteSequenceDataset(
        org_dirs=[os.path.join(args.data_path, 'input')],
        matte_dirs=[os.path.join(args.data_path, 'target')],
        background_dirs=[args.bg_path] if args.bg_path else None,
    )
    
    # Tạo một lambda function để truyền crop_sizes vào collate_fn
    full_resize_cfg = {
        'max_side': args.full_max_side,
        'max_pixels': args.full_max_pixels
    }
    collate_fn_with_args = lambda batch: custom_collate_fn(batch, args.crop_sizes, full_resize_cfg, args.background_mode)

    # Optimized DataLoader settings
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
        'collate_fn': collate_fn_with_args # Sử dụng collate_fn tùy chỉnh
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.visualize_dir, exist_ok=True) # Tạo thư mục visualize
    best_train_loss = float('inf')
    global_step = 0 # Biến đếm số batch trên toàn cục

    # ---- 2. Training Loop ----
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Dữ liệu đã được chuyển sang device trong custom_collate_fn
                input_9c = batch['input_9c'].to(device, non_blocking=True)
                image_curr = batch['image_curr'].to(device, non_blocking=True)
                matte_gt = batch['matte_gt'].to(device, non_blocking=True)
                background = batch['background'].to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast():
                    pred_alpha = model(input_9c)
                    
                    # Tính toán các thành phần loss
                    loss_alpha = l1_loss_fn(pred_alpha, matte_gt)
                    loss_comp = comp_loss_fn(pred_alpha, matte_gt, image_curr, background)
                    loss_lap = lap_loss_fn(pred_alpha, matte_gt)
                    
                    # Tổng loss có trọng số
                    total_loss = (args.weight_alpha * loss_alpha + 
                                  args.weight_comp * loss_comp +
                                  args.weight_lap * loss_lap)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Ghi log loss vào TensorBoard sau mỗi batch
                writer.add_scalar('Loss/train_total', total_loss.item(), global_step)
                writer.add_scalar('Loss/train_alpha', loss_alpha.item(), global_step)
                writer.add_scalar('Loss/train_comp', loss_comp.item(), global_step)
                writer.add_scalar('Loss/train_lap', loss_lap.item(), global_step)
                writer.add_scalar('Misc/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                global_step += 1

                if args.viz_every > 0 and global_step % args.viz_every == 0:
                    grid = create_comparison_grid(batch, pred_alpha.detach(), args.viz_samples)
                    writer.add_image('Realtime/Input-GT-Pred', grid, global_step)
                if args.viz_full_every > 0 and batch.get('meta') and global_step % args.viz_full_every == 0:
                    log_full_frame_preview(model, batch['meta'][0], writer, global_step, args, device)

                pbar.update(input_9c.size(0))
                epoch_loss += total_loss.item()
                pbar.set_postfix(**{'loss (batch)': total_loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        
        # Ghi loss trung bình của epoch
        epoch_avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch_avg', epoch_avg_loss, epoch)
        scheduler.step(epoch_avg_loss)

        # ---- 3. Checkpointing & Visualization ----
        if (epoch + 1) % args.save_every == 0:
            # Visualize output at the end of specified epochs
            visualize_output(batch, pred_alpha.detach(), epoch, args.visualize_dir, num_samples=args.viz_samples)
            
            log_images_to_tensorboard(writer, batch, pred_alpha.detach(), epoch, num_samples=args.viz_samples)

            print(f'\nTrain Loss (avg): {epoch_avg_loss:.4f}')

            if epoch_avg_loss < best_train_loss:
                best_train_loss = epoch_avg_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Saved best model to {checkpoint_path}')

            # Save periodic checkpoint
            periodic_checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), periodic_checkpoint_path)
            print(f'Saved periodic checkpoint to {periodic_checkpoint_path}')

    writer.close() # Đóng writer khi huấn luyện xong


def log_images_to_tensorboard(writer, batch, pred_alpha, epoch, num_samples=3):
    """Hàm chuyên dụng để ghi ảnh vào TensorBoard."""
    grid = create_comparison_grid(batch, pred_alpha, num_samples)
    writer.add_image('Comparison/Input-GT-Pred', grid, epoch)


def log_full_frame_preview(model, sample_meta, writer, global_step, args, device):
    """Chạy inference full frame và log lên TensorBoard để so với monitoring crops."""
    if not sample_meta:
        return

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_rgb(path):
        img = Image.open(path).convert('RGB')
        tensor = to_tensor(img)
        return tensor, normalize(tensor.clone())

    curr_vis, curr_norm = load_rgb(sample_meta['curr'])
    prev_vis, prev_norm = load_rgb(sample_meta['prev'])
    next_vis, next_norm = load_rgb(sample_meta['next'])
    matte_gt = to_tensor(Image.open(sample_meta['matte']).convert('L'))

    input_9c = torch.cat([prev_norm, curr_norm, next_norm], dim=0).unsqueeze(0).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred_alpha = inference_sliding_window(
            model,
            input_9c,
            tile_size=args.viz_tile_size,
            overlap=args.viz_tile_overlap,
            device=device
        ).cpu()
    if was_training:
        model.train()

    red_bg = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    gt_colored = red_bg * (1 - matte_gt)
    pred_colored = red_bg * (1 - pred_alpha.squeeze(0))

    grid = vutils.make_grid([curr_vis, gt_colored, pred_colored], nrow=3, padding=2, normalize=False)
    writer.add_image('Comparison/FullFrame', grid, global_step)


if __name__ == '__main__':
    args = get_args()
    train_model(args)
