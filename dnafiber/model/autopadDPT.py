from torch import nn


class AutoPad(nn.Module):
    def __init__(self, module, divisible_by=14):
        super().__init__()
        self.module = module
        self.divisible_by = divisible_by

    def forward(self, x):
        # Calculate padding to make input size divisible by divisible_by
        height, width = x.shape[2], x.shape[3]
        pad_h = self.divisible_by - height % self.divisible_by
        pad_w = self.divisible_by - width % self.divisible_by
        padding = (0, pad_w, 0, pad_h)
        x = nn.functional.pad(x, padding)
        # Forward pass through the module
        x = self.module(x)
        # Remove padding after the forward pass
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :height, :width]
        return x

    def __repr__(self):
        return f"AutoPad({self.module.__class__.__name__}"
