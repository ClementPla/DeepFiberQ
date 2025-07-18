from dnafiber.trainee import Trainee
from dnafiber.postprocess.fiber import FiberProps
import pandas as pd

def _get_model(revision, device="cuda"):
    if revision is None:
        model = Trainee.from_pretrained(
            "ClementP/DeepFiberQ", arch="unet", encoder_name="mit_b0"
        )
    else:
        model = Trainee.from_pretrained(
            "ClementP/DeepFiberQ",
            revision=revision,
        )
    return model.eval().to(device)


def format_results(results: list[FiberProps], pixel_size: float) -> pd.DataFrame:
    """
    Format the results for display in the UI.
    """
    results = [fiber for fiber in results if fiber.is_valid]
    all_results = dict(
        FirstAnalog=[], SecondAnalog=[], length=[], ratio=[], fiber_type=[]
    )
    all_results["FirstAnalog"].extend([fiber.red * pixel_size for fiber in results])
    all_results["SecondAnalog"].extend([fiber.green * pixel_size for fiber in results])
    all_results["length"].extend(
        [fiber.red * pixel_size + fiber.green * pixel_size for fiber in results]
    )
    all_results["ratio"].extend([fiber.ratio for fiber in results])
    all_results["fiber_type"].extend([fiber.fiber_type for fiber in results])

    return pd.DataFrame.from_dict(all_results)




MODELS_ZOO = {
    "Ensemble": "ensemble",
    "SegFormer MiT-B4": "segformer_mit_b4",
    "SegFormer MiT-B2": "segformer_mit_b2",
    "U-Net SE-ResNet50": "unet_se_resnet50",
}