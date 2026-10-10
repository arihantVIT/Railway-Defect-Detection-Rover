from ultralytics import YOLO
from IPython.display import display, Image
from pathlib import Path

# Load your trained model
model = YOLO("/home/arihant/Desktop/AI_ML/ESA_ROVER_1/controllers/railway_rover/best.pt")

# Path to the test image
image_path = "/home/arihant/Desktop/AI_ML/ESA_ROVER_1/controllers/railway_rover/img.png"

# Run inference
results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    save=True,
    show=False
)

# Display the detection results
result = results[0]
display(Image(filename=result.save_dir and str(Path(result.save_dir) / Path(image_path).name)))
