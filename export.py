import argparse
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.torch_utils import select_device
from export import run  # Import the export function

# List of models to export
MODEL_NAMES = ['drone', 'bird', 'aeroplane']

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-dir', type=str, default='weights/', help='Directory where trained models are stored')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for export')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--device', default='cpu', help='Device to use for inference (cpu or cuda)')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'engine'], 
                        help='Formats to export: torchscript, onnx, coreml, engine, tflite')
    opt = parser.parse_args()
    return opt

def main(opt):
    device = select_device(opt.device)
    
    for model_name in MODEL_NAMES:
        model_path = Path(opt.weights_dir) / f"{model_name}.pt"
        
        if not model_path.exists():
            print(f"⚠️ Model file {model_path} not found, skipping...")
            continue

        print(f"🔄 Exporting {model_name}.pt...")
        
        run(weights=str(model_path),
            imgsz=opt.img_size,
            batch_size=opt.batch_size,
            device=device,
            include=opt.include)  # Export in specified formats
        
        print(f"✅ Export completed for {model_name}.pt")

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
