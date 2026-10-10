# Railway Rover Defect Detection

This project is a Webots-based railway inspection rover that uses a YOLO object detection model to detect railway defects in real time from a robot camera feed. Once a defect is detected, the rover pauses, logs the event, and resumes movement after a short delay.

## Overview

The system combines:

- Webots simulation for the railway environment
- A custom rover controller written in Python
- Ultralytics YOLO for real-time defect detection
- GPS and motion control for rover navigation
- CSV logging for defect events and confidence scores

## Project structure

```text
ESA_ROVER_1/
├── controllers/
│   └── railway_rover/
│       ├── best.pt                     # Trained YOLO model
│       ├── railway_rover.py           # Main rover controller
│       ├── code.py                    # Quick inference sample
│       ├── railway_defect_log.csv     # Logged detections
│       ├── dataset/                   # Dataset files / labeling data
│       └── runtime.ini                # Python runtime for Webots
├── worlds/
│   ├── railway_track.wbt
│   ├── railway_track_claude.wbt
│   └── railway_track (Copy).wbt
├── railway_env/
│   └── bin/python                    # Virtual environment python
├── runs/
│   └── detect/
│       └── predict/
├── Model_Training/
├── plugins/
├── libraries/
├── protos/
└── README.md
```

## Main controller

The main logic lives in:

- `controllers/railway_rover/railway_rover.py`

This script:

- initializes the robot and camera
- loads the trained YOLO model from `best.pt`
- captures frames from the onboard camera
- runs inference with a confidence threshold
- selects the highest-confidence defect detection
- pauses the rover for a short stop period
- stores defect details into `railway_defect_log.csv`
- resumes robot motion after the pause

## Requirements

- Python 3.10
- Webots
- A virtual environment with the required Python packages

Recommended environment:

```bash
cd /home/arihant/Desktop/AI_ML/ESA_ROVER_1
source railway_env/bin/activate
```

If needed, install the required dependencies:

```bash
pip install ultralytics opencv-python numpy
```

## Quick start

1. Activate the project environment:

```bash
cd /home/arihant/Desktop/AI_ML/ESA_ROVER_1
source railway_env/bin/activate
```

2. Open one of the world files in Webots:

```text
worlds/railway_track.wbt
```

3. Assign the rover controller to the robot in the simulation.

4. Start the simulation.

The controller will automatically load the model and begin defect detection.

## Runtime behavior

The controller is configured with:

- `SPEED = 10.0`
- `CONFIDENCE_THRESHOLD = 0.25`
- `IMAGE_SIZE = 640`
- `DETECTION_DELAY = 2`
- `STOP_DURATION = 2.0`

This means the rover keeps moving until it sees a valid defect above the confidence threshold, then it pauses briefly before resuming.

## Logging

Detected defects are saved to:

- `controllers/railway_rover/railway_defect_log.csv`

Columns include:

- defect category
- confidence value
- detection start time
- GPS position
- stop time

## Model and dataset notes

- `controllers/railway_rover/best.pt` is the trained detection model.
- `controllers/railway_rover/dataset/` contains image data used for training or validation.
- `code.py` is a lightweight sample script for running model inference on a single image.

## Useful commands

Run a quick model inference example:

```bash
cd /home/arihant/Desktop/AI_ML/ESA_ROVER_1
source railway_env/bin/activate
python controllers/railway_rover/code.py
```

## Notes

- The project is designed for a Webots simulation workflow rather than a standalone desktop app.
- `runtime.ini` already points to the project virtual environment Python interpreter for Webots integration.
- You can replace the current model with a retrained version by updating `best.pt` and restarting the controller.

## Future improvements

Possible next steps include:

- improving defect classification coverage
- adding event-based visualization overlays
- storing more detailed rover telemetry
- creating automatic retraining pipelines from the dataset
