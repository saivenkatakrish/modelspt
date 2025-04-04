import argparse
import os
from pathlib import Path

import torch
from train import train  # Import YOLOv5 training function

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['drone', 'bird', 'aeroplane'], required=True,
                        help='Dataset to train on: drone, bird, or aeroplane')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Pretrained weights path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    opt = parser.parse_args()
    return opt

def main(opt):
    dataset_name = opt.dataset
    dataset_path = f'data/{dataset_name}.yaml'  # Path to dataset YAML file
    save_model_path = f'weights/{dataset_name}.pt'  # Output model path
    
    train(hyp='data/hyp.scratch.yaml',  # Default hyperparameters
          data=dataset_path,
          weights=opt.weights,
          epochs=opt.epochs,
          batch_size=opt.batch_size,
          imgsz=opt.img_size,
          device=opt.device,
          project='runs/train',
          name=dataset_name,
          exist_ok=True)
    
    print(f'Training completed. Model saved as {save_model_path}')

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
