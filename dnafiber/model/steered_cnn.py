from steered_cnn.models import SteeredUNet
from kornia.filters import gaussian_blur2d, spatial_gradient
import torch


class DNAFiberSteeredCNN(SteeredUNet):
    def __init__(self, arch, encoder_name, n_in=3, classes=3, dropout=0.0, **kwargs):
        super().__init__(n_in=n_in, n_out=classes, p_dropout=dropout, **kwargs)

    def forward(self, x):
        """
        Forward pass through the Steered CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the model.
        """

        # Compute an estimate of the angle based on Laplacian of Gaussian of the input tensor.
        angle = self.compute_angle(x)

        yhat = super().forward(x, angle)
        return yhat

    def compute_angle(self, x):
        """
        Compute an estimate of the angle based on the gradient of the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Estimated angle.
        """

        gray = x.max(dim=1, keepdim=True)[
            0
        ]  # Convert to grayscale by taking the max across channels

        blurred = gaussian_blur2d(gray, (5, 5), (1.0, 1.0))

        edges = spatial_gradient(blurred, normalized=True)

        gx = edges[:, :, 0]
        gy = edges[:, :, 1]
        angle = torch.atan2(gy, gx)  # Compute the angle from the gradient

        return angle.squeeze(1)  # Remove the channel dimension for compatibility
