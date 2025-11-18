import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaL1Loss(nn.Module):
    def __init__(self):
        super(AlphaL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_alpha, gt_alpha, mask=None):
        if mask is not None:
            pred_alpha = pred_alpha * mask
            gt_alpha = gt_alpha * mask
        return self.l1_loss(pred_alpha, gt_alpha)

class CompositionalLoss(nn.Module):
    def __init__(self):
        super(CompositionalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_alpha, gt_alpha, original_image, random_background, mask=None):
        """
        original_image: The original frame at time t (img_curr_t)
        random_background: A random background image (img_bg_t)
        """
        
        # Recreate the image with the predicted alpha
        pred_composed = original_image * pred_alpha + random_background * (1.0 - pred_alpha)
        
        # Recreate the image with the ground truth alpha
        # Note: The guide uses original_image here, which assumes the background is known.
        # A more robust approach for pre-keyed footage is to estimate foreground F and compose.
        # F_gt = original_image / (gt_alpha + 1e-6)
        # gt_composed = F_gt * gt_alpha + random_background * (1.0 - gt_alpha)
        # However, we stick to the guide's simpler version.
        gt_composed = original_image * gt_alpha + random_background * (1.0 - gt_alpha)
        
        if mask is not None:
            pred_composed = pred_composed * mask
            gt_composed = gt_composed * mask

        return self.l1_loss(pred_composed, gt_composed)

class GradientLoss(nn.Module):
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
        # Define Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        
        # Register as non-trainable parameters
        self.sobel_x = nn.Parameter(sobel_x, requires_grad=False).to(device)
        self.sobel_y = nn.Parameter(sobel_y, requires_grad=False).to(device)

    def forward(self, pred_alpha, gt_alpha, mask=None):
        # Ensure input is single channel
        if pred_alpha.shape[1] != 1:
            pred_alpha = pred_alpha.mean(dim=1, keepdim=True)
        if gt_alpha.shape[1] != 1:
            gt_alpha = gt_alpha.mean(dim=1, keepdim=True)

        # Calculate gradients for prediction
        pred_grad_x = F.conv2d(pred_alpha, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_alpha, self.sobel_y, padding=1)
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
        # Calculate gradients for ground truth
        gt_grad_x = F.conv2d(gt_alpha, self.sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt_alpha, self.sobel_y, padding=1)
        gt_grad_mag = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-8)
        
        if mask is not None:
            pred_grad_mag = pred_grad_mag * mask
            gt_grad_mag = gt_grad_mag * mask

        return self.l1_loss(pred_grad_mag, gt_grad_mag)

class LaplacianPyramidLoss(nn.Module):
    def __init__(self, max_levels=5, device='cuda'):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.l1_loss = nn.L1Loss()
        
        # Create Gaussian kernel
        kernel = torch.tensor([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
        self.kernel = nn.Parameter(kernel.view(1, 1, 5, 5), requires_grad=False).to(device)

    def gaussian_blur(self, x):
        # Ensure kernel is on the same device as x
        if self.kernel.device != x.device:
            self.kernel = self.kernel.to(x.device)
        return F.conv2d(x, self.kernel, padding=2)

    def build_pyramid(self, x):
        pyramid = []
        current = x
        for _ in range(self.max_levels):
            blurred = self.gaussian_blur(current)
            laplacian = current - blurred
            pyramid.append(laplacian)
            current = F.avg_pool2d(blurred, 2)
        # The guide is ambiguous here, but typically the last blurred image is also added.
        # pyramid.append(current) 
        return pyramid

    def forward(self, pred_alpha, gt_alpha, mask=None):
        pred_pyramid = self.build_pyramid(pred_alpha)
        gt_pyramid = self.build_pyramid(gt_alpha)
        
        loss = 0
        for i in range(self.max_levels):
            # The weight 2**i is mentioned in the guide, but often just summing the L1 is effective.
            # We will follow the guide's formula.
            weight = 2**i
            
            if mask is not None:
                # Resize mask for each level
                mask_i = F.interpolate(mask, size=pred_pyramid[i].shape[2:], mode='nearest')
                loss += weight * self.l1_loss(pred_pyramid[i] * mask_i, gt_pyramid[i] * mask_i)
            else:
                loss += weight * self.l1_loss(pred_pyramid[i], gt_pyramid[i])
                
        return loss
