import torch
import torch.nn as nn

def make_model(args):
    return MSRN(args)

class MSRB(nn.Module):
    """Multi-scale Residual Block"""
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()

        # Convolutions with different kernel sizes to capture multi-scale features
        self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.conv_5_1 = nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2)
        
        # The output of the two branches are concatenated, so the input to the next conv is 2 * n_feats
        self.conv_3_2 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1)
        self.conv_5_2 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2)

        # Bottleneck convolution to merge features
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input residual
        res = x

        # First level of multi-scale convolutions
        out_3_1 = self.relu(self.conv_3_1(x))
        out_5_1 = self.relu(self.conv_5_1(x))
        
        # Concatenate the outputs
        input_2 = torch.cat([out_3_1, out_5_1], 1)

        # Second level of multi-scale convolutions
        out_3_2 = self.relu(self.conv_3_2(input_2))
        out_5_2 = self.relu(self.conv_5_2(input_2))

        # Concatenate all feature maps
        out = torch.cat([out_3_2, out_5_2], 1)
        
        # Apply bottleneck and add residual
        out = self.confusion(out)
        out += res
        
        return out

class MSRN(nn.Module):
    """Multi-scale Residual Network for Alpha Matting"""
    def __init__(self, n_channels_in=9, n_classes_out=1, n_feats=64, n_blocks=8):
        super(MSRN, self).__init__()
        
        # Shallow feature extraction
        self.conv_in = nn.Conv2d(n_channels_in, n_feats, kernel_size=3, padding=1)
        
        # Main body of the network: a sequence of MSRB blocks
        modules_body = [MSRB(n_feats=n_feats) for _ in range(n_blocks)]
        self.body = nn.Sequential(*modules_body)
        
        # Reconstruction layer
        self.conv_out = nn.Conv2d(n_feats, n_classes_out, kernel_size=3, padding=1)

    def forward(self, x):
        # Extract shallow features
        x = self.conv_in(x)
        
        # Pass through the main body
        x = self.body(x)
        
        # Reconstruct the image
        logits = self.conv_out(x)
        
        # Apply clamp activation for alpha matte
        return torch.clamp(logits, 0, 1)
