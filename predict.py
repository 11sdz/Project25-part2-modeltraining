import cv2
from ultralytics import YOLO
import torch

# Load the model
model = YOLO('runs/segment/train6/weights/best.pt')  # Adjust your path

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open video file
cap = cv2.VideoCapture('DJI0008.mp4')
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_predicted.mp4', fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on the frame
    results = model.predict(frame, device=device, verbose=False)

    # Get the annotated frame (you may have to adjust depending on model output type)
    annotated_frame = results[0].plot()  # returns a NumPy image

    # Write to output video
    out.write(annotated_frame)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

# Clean up
cap.release()
out.release()
print("Prediction complete. Saved to 'output_predicted.mp4'")
