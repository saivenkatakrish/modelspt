import sys
from pathlib import Path

# Automatically add the directory where 'val.py' is located to sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory where val.py is located
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Now import YOLOv5 modules
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# Set device
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths (adjust if needed)
model_paths = {
    "drone": "modelspt/drone.pt",
    "bird": "modelspt/bird.pt",
    "aeroplane": "modelspt/aeroplane.pt"
}

# Load models
models = {name: DetectMultiBackend(path, device=device) for name, path in model_paths.items()}

# Print model summaries
for name, model in models.items():
    print(f"{name} model loaded successfully!")
