import torch
import pathlib
from pathlib import Path
import numpy as np
from PIL import Image
import argparse

# Conditional patch for Windows compatibility
if isinstance(Path(), pathlib.WindowsPath):
    pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 model loading and inference utilities
from models.common import AutoShape, DetectMultiBackend
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel, SegmentationModel
from utils.downloads import attempt_download
from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging, cv2, print_args
from utils.torch_utils import select_device


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    if not verbose:
        LOGGER.setLevel(logging.WARNING)

    check_requirements(ROOT / "requirements.txt", exclude=("opencv-python", "tensorboard", "thop"))
    name = Path(name)
    path = name.with_suffix(".pt") if name.suffix == "" and not name.is_dir() else name  # checkpoint path

    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 80:
            try:
                model = DetectMultiBackend(path, device=device, fuse=autoshape)
                if autoshape:
                    if model.pt and isinstance(model.model, ClassificationModel):
                        LOGGER.warning(
                            "⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. "
                            "Use input shape (1, 3, H, W) with torch tensors."
                        )
                    elif model.pt and isinstance(model.model, SegmentationModel):
                        LOGGER.warning(
                            "⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. "
                            "You will not be able to run inference with this model."
                        )
                    else:
                        model = AutoShape(model)
            except Exception:
                model = attempt_load(path, device=device, fuse=False)
        else:
            cfg = list((Path(__file__).parent / "models").rglob(f"{path.stem}.yaml"))[0]
            model = DetectionModel(cfg, channels, classes)
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)
                csd = ckpt["model"].float().state_dict()
                csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])
                model.load_state_dict(csd, strict=False)
                if len(ckpt["model"].names) == classes:
                    model.names = ckpt["model"].names
        if not verbose:
            LOGGER.setLevel(logging.INFO)
        return model.to(device)

    except Exception as e:
        help_url = "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading"
        raise Exception(f"{e}. Cache may be out of date. Try `force_reload=True` or see {help_url}") from e


# Custom loader for custom.pt
def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None):
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device)

# Predefined model functions
def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5n", pretrained, channels, classes, autoshape, _verbose, device)

def best(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("best", pretrained, channels, classes, autoshape, _verbose, device)

def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device)

def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5l", pretrained, channels, classes, autoshape, _verbose, device)

def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5x", pretrained, channels, classes, autoshape, _verbose, device)

def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5n6", pretrained, channels, classes, autoshape, _verbose, device)

def best6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("best6", pretrained, channels, classes, autoshape, _verbose, device)

def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5m6", pretrained, channels, classes, autoshape, _verbose, device)

def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5l6", pretrained, channels, classes, autoshape, _verbose, device)

def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None):
    return _create("yolov5x6", pretrained, channels, classes, autoshape, _verbose, device)


# CLI Testing Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best", help="model name or path to custom model.pt")
    opt = parser.parse_args()
    print_args(vars(opt))

    # Load model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)

    # Test Images (support file, path, URL, OpenCV, PIL, NumPy)
    imgs = [
        "data/images/zidane.jpg",
        Path("data/images/zidane.jpg"),
        "https://ultralytics.com/images/zidane.jpg",
        cv2.imread("data/images/bus.jpg")[:, :, ::-1],
        Image.open("data/images/bus.jpg"),
        np.zeros((320, 640, 3)),
    ]

    # Inference
    results = model(imgs, size=320)

    # Output
    results.print()
    results.save()
