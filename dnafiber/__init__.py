import types
import torch
import warnings

warnings.filterwarnings(
    "ignore", message="Torchaudio's I/O functions now support par-call"
)
# Fix torch.classes for Streamlit's file watcher
torch.classes.__path__ = types.SimpleNamespace(_path=[])
