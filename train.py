import torch
from ultralytics import YOLO

def main():
    # Check PyTorch version and CUDA availability
    print(torch.__version__)  # Check PyTorch version
    print(torch.cuda.is_available())  # Should return True if CUDA is available

    # Check if CUDA is available and set the device to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print the device being used
    print(f"Using device: {device}")

    # Load a pre-trained YOLOv11n model and move it to the GPU
    model = YOLO('yolo11n.pt').to(device)

    # Start training on the GPU
    results = model.train(data='data.yaml', epochs=100, imgsz=640, device=device)

if __name__ == '__main__':
    main()
