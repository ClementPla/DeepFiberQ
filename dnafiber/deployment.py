from huggingface_hub import PyTorchModelHubMixin
import torch
from segmentation_models_pytorch import create_model


class UnetMIT_B0(PyTorchModelHubMixin, torch.nn.Module):
    def __init__(self, model=None):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            self.model = create_model(
                arch="unet",
                encoder_name="mit_b0",
                classes=3,
            )


def get_model():
    return UnetMIT_B0.from_pretrained("ClementP/DeepFiberQ")
