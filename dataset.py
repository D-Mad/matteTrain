import os
import random
from torch.utils.data import Dataset

class AlphaMatteSequenceDataset(Dataset):
    def __init__(self, org_dirs, matte_dirs, background_dirs=None):
        """
        Dataset giờ đây chỉ chịu trách nhiệm tải đường dẫn file.
        Tất cả logic crop và transform sẽ được xử lý trong collate_fn.
        """
        self.samples = []
        self.background_images = []

        # --- PHẦN 1: Xây dựng chỉ mục (Giữ nguyên code cũ của bạn) ---
        print("Building dataset index...")
        for org_dir, matte_dir in zip(org_dirs, matte_dirs):
            frames = sorted([f for f in os.listdir(org_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            num_frames = len(frames)
            
            for i in range(num_frames):
                frame_name = frames[i]
                curr_frame_path = os.path.join(org_dir, frame_name)
                
                matte_frame_path = None
                base_name, _ = os.path.splitext(frame_name)
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_matte_path = os.path.join(matte_dir, base_name + ext)
                    if os.path.exists(potential_matte_path):
                        matte_frame_path = potential_matte_path
                        break
                
                prev_frame_path = os.path.join(org_dir, frames[max(i - 1, 0)])
                next_frame_path = os.path.join(org_dir, frames[min(i + 1, num_frames - 1)])

                if not (os.path.exists(curr_frame_path) and matte_frame_path and os.path.exists(matte_frame_path)):
                    print(f"Warning: Skipping frame {curr_frame_path} due to missing matte.")
                    continue 

                self.samples.append({
                    'prev': prev_frame_path,
                    'curr': curr_frame_path,
                    'next': next_frame_path,
                    'matte': matte_frame_path
                })
        
        if background_dirs:
            for bg_dir in background_dirs:
                if not os.path.isdir(bg_dir):
                    continue
                self.background_images.extend([
                    os.path.join(bg_dir, f)
                    for f in os.listdir(bg_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])

        print(f"Dataset index built. Found {len(self.samples)} samples.")
        print(f"Background images: {len(self.background_images)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        bg_path = None
        if self.background_images:
            bg_path = random.choice(self.background_images)

        # Chỉ trả về đường dẫn, việc tải ảnh sẽ do collate_fn xử lý
        return {
            'paths': sample_info,
            'bg_path': bg_path
        }
