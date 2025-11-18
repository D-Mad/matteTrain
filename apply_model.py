import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import argparse

from unet_model import UNet9C
from res_unet_model import ResUNet

def get_args():
    parser = argparse.ArgumentParser(description='Apply a trained model to an image sequence.')
    parser.add_argument('--model-path', '-m', type=str, required=True, help='Path to the trained model (.pth file)')
    parser.add_argument('--input-dir', '-i', type=str, required=True, help='Directory containing the input image sequence')
    parser.add_argument('--output-dir', '-o', type=str, default='output_mattes', help='Directory to save the output alpha mattes')
    parser.add_argument('--model-type', type=str, default='resunet', choices=['resunet', 'unet'], help='Type of model architecture to use')
    # Thêm tham số cho Sliding Window Inference
    parser.add_argument('--tile-size', type=int, default=512, help='The size of each tile for sliding window inference.')
    parser.add_argument('--tile-overlap', type=int, default=64, help='The overlap between tiles for sliding window inference.')
    return parser.parse_args()

def inference_sliding_window(model, input_tensor, tile_size, overlap, device):
    """
    Thực hiện inference trên ảnh lớn bằng kỹ thuật sliding window (tiled inference).
    """
    B, C, H, W = input_tensor.shape
    
    # Kích thước bước nhảy của cửa sổ
    stride = tile_size - overlap
    
    # Tạo các tensor để tích lũy kết quả và trọng số pha trộn
    output_matte = torch.zeros((B, 1, H, W), device=device)
    weight_map = torch.zeros((B, 1, H, W), device=device)

    # Tạo cửa sổ pha trộn (blending window) để làm mượt các đường nối
    # Sử dụng cửa sổ tuyến tính (Bartlett) cho đơn giản
    blend_window = torch.bartlett_window(tile_size, periodic=False, device=device)
    blend_window_2d = torch.outer(blend_window, blend_window).unsqueeze(0).unsqueeze(0)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Xác định tọa độ của ô (tile)
            y_start, y_end = y, min(y + tile_size, H)
            x_start, x_end = x, min(x + tile_size, W)
            
            # Lấy ô từ ảnh đầu vào
            tile = input_tensor[:, :, y_start:y_end, x_start:x_end]
            
            # Nếu kích thước ô nhỏ hơn tile_size, cần đệm (pad) để model nhận đúng kích thước
            h_tile, w_tile = tile.shape[2:]
            pad_h = tile_size - h_tile
            pad_w = tile_size - w_tile
            if pad_h > 0 or pad_w > 0:
                # For small tiles (e.g. image smaller than tile_size) reflection padding fails,
                # so fall back to constant padding in those cases.
                if (pad_h > 0 and pad_h >= h_tile) or (pad_w > 0 and pad_w >= w_tile):
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='constant', value=0)
                else:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            # Chạy inference trên ô
            with torch.no_grad():
                pred_tile = model(tile)
            
            # Cắt bỏ phần đệm nếu có
            pred_tile = pred_tile[:, :, :h_tile, :w_tile]
            
            # Lấy cửa sổ pha trộn tương ứng với kích thước ô
            current_blend_window = blend_window_2d[:, :, :h_tile, :w_tile]

            # Cộng dồn kết quả và trọng số vào các bản đồ lớn
            output_matte[:, :, y_start:y_end, x_start:x_end] += pred_tile * current_blend_window
            weight_map[:, :, y_start:y_end, x_start:x_end] += current_blend_window

    # Chuẩn hóa kết quả bằng cách chia cho tổng trọng số
    # Thêm một epsilon nhỏ để tránh chia cho 0
    final_matte = output_matte / (weight_map + 1e-8)
    
    return final_matte.clamp(0, 1)


def apply_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model based on type
    if args.model_type == 'resunet':
        print("Loading ResUNet model.")
        model = ResUNet(n_channels_in=9, n_classes_out=1)
    else: # 'unet'
        print("Loading UNet9C model.")
        model = UNet9C(n_channels_in=9, n_classes_out=1)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Data transformation without resizing
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Khôi phục lại bước chuẩn hóa để khớp với quy trình huấn luyện mới
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get image list
    image_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"No images found in {args.input_dir}")
        return

    # Helper to load and transform frames on demand
    def load_frame_tensor(frame_idx):
        frame_path = os.path.join(args.input_dir, image_files[frame_idx])
        with Image.open(frame_path).convert('RGB') as img:
            return transform(img)

    # Preload the first two frames so prev/curr/next match training setup
    prev_tensor = load_frame_tensor(0)
    curr_tensor = prev_tensor
    next_tensor = load_frame_tensor(1) if len(image_files) > 1 else curr_tensor

    print("Processing image sequence with Sliding Window Inference...")
    with torch.no_grad():
        for i, frame_name in enumerate(tqdm(image_files, unit='frame')):
            # Stack prev/curr/next just like in training
            input_9c = torch.cat([prev_tensor, curr_tensor, next_tensor], dim=0).unsqueeze(0).to(device)

            # Predict using sliding window
            pred_alpha = inference_sliding_window(
                model, 
                input_9c, 
                tile_size=args.tile_size, 
                overlap=args.tile_overlap, 
                device=device
            )

            # Convert to image and save
            # The prediction is for the center frame of the buffer
            output_img = transforms.ToPILImage()(pred_alpha.squeeze(0).cpu())
            
            save_path = os.path.join(args.output_dir, frame_name)
            output_img.save(save_path)

            # Prepare tensors for the next frame so that the temporal context always matches training
            if i < len(image_files) - 1:
                prev_tensor = curr_tensor
                curr_tensor = next_tensor
                if i + 2 < len(image_files):
                    next_tensor = load_frame_tensor(i + 2)
                else:
                    next_tensor = curr_tensor

    print(f"Inference complete. Mattes saved to {args.output_dir}")

if __name__ == '__main__':
    args = get_args()
    apply_model(args)
