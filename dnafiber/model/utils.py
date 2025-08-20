from dotenv import load_dotenv

load_dotenv()
import os

from huggingface_hub import HfApi
from lightning.pytorch.utilities import rank_zero_only

HF_TOKEN = os.environ.get("HF_TOKEN")


@rank_zero_only
def upload_to_hub(model, arch, encoder):
    hfapi = HfApi()
    branch_name = f"{arch}_{encoder}"
    hfapi.create_repo(
        "ClementP/DeepFiberQV2",
        token=HF_TOKEN,
        exist_ok=True,
        repo_type="model",
    )
    hfapi.create_branch(
        "ClementP/DeepFiberQV2",
        branch=branch_name,
        token=HF_TOKEN,
        exist_ok=True,
    )

    model.push_to_hub(
        "ClementP/DeepFiberQV2",
        branch=branch_name,
        token=HF_TOKEN,
    )
