import argparse
import torch
from pathlib import Path
from val import run  # Import YOLOv5 validation function

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['drone', 'bird', 'aeroplane'], required=True,
                        help='Dataset to validate: drone, bird, or aeroplane')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (overrides default)')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for validation')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to validate on')
    opt = parser.parse_args()
    return opt

def main(opt):
    dataset_name = opt.dataset
    dataset_path = f'data/{dataset_name}.yaml'  # Path to dataset YAML file
    model_weights = opt.weights if opt.weights else f'weights/{dataset_name}.pt'  # Select model weights
    
    run(data=dataset_path,
        weights=model_weights,
        imgsz=opt.img_size,
        batch_size=opt.batch_size,
        device=opt.device)
    
    print(f'Validation completed for {dataset_name} dataset using {model_weights}')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
