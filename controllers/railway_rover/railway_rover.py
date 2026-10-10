from controller import Robot
from ultralytics import YOLO
import cv2
import numpy as np
import csv
import os

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = (
    "/home/arihant/Desktop/AI_ML/"
    "ESA_ROVER_1/controllers/railway_rover/best.pt"
)

SPEED = 10.0
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Time to move after first detecting a defect
DETECTION_DELAY = 2

# Time to remain stopped before resuming
STOP_DURATION = 2.0

# CSV file saved in the controller directory
CSV_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "railway_defect_log.csv"
)

# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("Model classes:", model.names)

# ==========================================
# INITIALIZE WEBOTS
# ==========================================

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ==========================================
# CAMERA
# ==========================================

camera = robot.getDevice("cam")
camera.enable(timestep)

# ==========================================
# GPS
# ==========================================

gps = robot.getDevice("gps")
gps.enable(timestep)

# ==========================================
# MOTORS
# ==========================================

motor_fr = robot.getDevice("motor-1")
motor_rr = robot.getDevice("motor-4")
motor_rl = robot.getDevice("motor-3")
motor_fl = robot.getDevice("motor_2")

motors = [
    motor_fr,
    motor_rr,
    motor_rl,
    motor_fl
]

for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# ==========================================
# MOTOR FUNCTIONS
# ==========================================

def move_forward(speed):
    for motor in motors:
        motor.setVelocity(speed)


def stop_movement():
    for motor in motors:
        motor.setVelocity(0.0)

# ==========================================
# CSV INITIALIZATION
# ==========================================

if not os.path.exists(CSV_FILE):

    with open(CSV_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "defect_category",
            "confidence",
            "t_start",
            "gps_x",
            "gps_y",
            "gps_z",
            "stop_time"
        ])

print("Defect log:", CSV_FILE)

# ==========================================
# DETECTION STATE
# ==========================================

detection_active = False
detection_recorded = False

t_start = None
defect_category = None
defect_confidence = 0.0

stop_started_at = None
waiting_to_resume = False

# ==========================================
# MAIN LOOP
# ==========================================

print("==========================================")
print("RAILWAY DEFECT DETECTION STARTED")
print("==========================================")
print("Detection delay:", DETECTION_DELAY, "seconds")
print("Stop duration:", STOP_DURATION, "seconds")

try:

    while robot.step(timestep) != -1:

        current_time = robot.getTime()

        # ==================================
        # STATE 1: STOPPED AFTER DETECTION
        # ==================================

        if waiting_to_resume:

            stop_movement()

            # Resume after the configured pause
            if (
                current_time - stop_started_at
                >= STOP_DURATION
            ):

                waiting_to_resume = False

                # Reset detection state for the next event
                detection_active = False
                detection_recorded = False

                t_start = None
                defect_category = None
                defect_confidence = 0.0

                print("\nResuming rover movement...\n")

            else:

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested.")
                    break

                continue

        # ==================================
        # GET CAMERA FRAME
        # ==================================

        raw = camera.getImage()

        if not raw:
            move_forward(SPEED)
            continue

        frame = np.frombuffer(
            raw,
            dtype=np.uint8
        ).reshape(
            (
                camera.getHeight(),
                camera.getWidth(),
                4
            )
        )

        # Convert BGRA to BGR
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )

        # ==================================
        # YOLO INFERENCE
        # ==================================

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        # Annotated frame for visualization
        annotated_frame = result.plot()

        # ==================================
        # PROCESS DETECTIONS
        # ==================================

        detected = False
        current_category = None
        current_confidence = 0.0

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            # Select the highest-confidence detection
            confidences = (
                result.boxes.conf.cpu().numpy()
            )

            best_index = int(
                np.argmax(confidences)
            )

            current_confidence = float(
                confidences[best_index]
            )

            class_id = int(
                result.boxes.cls[best_index].item()
            )

            current_category = model.names[class_id]

            detected = (
                current_confidence
                >= CONFIDENCE_THRESHOLD
            )

        # ==================================
        # FIRST DETECTION: RECORD t_start
        # ==================================

        if detected and not detection_active:

            detection_active = True

            t_start = current_time

            defect_category = current_category
            defect_confidence = current_confidence

            print("\n==========================================")
            print("DEFECT DETECTED")
            print("==========================================")

            print("Category:", defect_category)

            print(
                f"Confidence: {defect_confidence:.3f}"
            )

            print(
                f"Detection start time: {t_start:.3f} s"
            )

            print(
                f"Stopping in approximately "
                f"{DETECTION_DELAY:.1f} seconds."
            )

        # ==================================
        # WAIT FOR DETECTION DELAY
        # ==================================

        if (
            detection_active
            and not detection_recorded
        ):

            elapsed = current_time - t_start

            if elapsed >= DETECTION_DELAY:

                # Stop all motors
                stop_movement()

                detection_recorded = True
                waiting_to_resume = True
                stop_started_at = current_time

                # Read GPS coordinates
                position = gps.getValues()

                x = float(position[0])
                y = float(position[1])
                z = float(position[2])

                # ==================================
                # PRINT DEFECT REPORT
                # ==================================

                print("\n")
                print("==========================================")
                print("          RAILWAY DEFECT REPORT")
                print("==========================================")

                print(
                    "Defect category:",
                    defect_category
                )

                print(
                    f"Confidence: {defect_confidence:.3f}"
                )

                print(
                    f"Detection start: {t_start:.3f} s"
                )

                print(
                    f"GPS X: {x:.3f} m"
                )

                print(
                    f"GPS Y: {y:.3f} m"
                )

                print(
                    f"GPS Z: {z:.3f} m"
                )

                print(
                    f"Rover stop time: "
                    f"{stop_started_at:.3f} s"
                )

                print("------------------------------------------")
                print("Defect recorded successfully.")
                print("Rover stopped.")
                print("==========================================")

                # ==================================
                # SAVE TO CSV
                # ==================================

                with open(
                    CSV_FILE,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        defect_category,
                        defect_confidence,
                        t_start,
                        x,
                        y,
                        z,
                        stop_started_at
                    ])

                print("Saved to:", CSV_FILE)

                print(
                    f"Waiting {STOP_DURATION:.1f} seconds "
                    f"before resuming."
                )

        # ==================================
        # DISPLAY CAMERA
        # ==================================

        cv2.imshow(
            "Railway Rover - YOLO Detection",
            annotated_frame
        )

        # ==================================
        # QUIT USING Q
        # ==================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("Quit requested. Stopping rover.")
            break

        # ==================================
        # MOVE FORWARD
        # ==================================

        if not waiting_to_resume:
            move_forward(SPEED)

finally:

    # Always stop motors when controller exits
    stop_movement()

    cv2.destroyAllWindows()

    print("==========================================")
    print("Rover stopped.")
    print("==========================================")