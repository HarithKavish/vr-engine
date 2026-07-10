from ultralytics import YOLO
import cv2
import numpy as np
import time

# ----------------------------------------
# Load YOLO11 Pose Model
# Automatically downloads on first run
# ----------------------------------------

MODEL_NAME = "yolo11n-pose.pt"

print("=" * 50)
print("VR Engine V1")
print(f"Loading model: {MODEL_NAME}")
print("If the model is not found, it will be downloaded automatically.")
print("=" * 50)

model = YOLO(MODEL_NAME)

# ----------------------------------------
# Camera
# ----------------------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window = "Harith VR Tracking"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

prev = time.time()

# ----------------------------------------
# Skeleton Connections
# ----------------------------------------

SKELETON = [

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16)

]

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Black canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    start = time.time()

    # GPU inference
    results = model(
        frame,
        device=0,
        verbose=False
    )

    end = time.time()

    fps = 1 / (time.time() - prev)
    prev = time.time()

    inference = (end - start) * 1000

    speed = results[0].speed

    if len(results):

        result = results[0]

        if result.keypoints is not None:

            points = result.keypoints.xy.cpu().numpy()[0]
            confidence = result.keypoints.conf.cpu().numpy()[0]

            # Draw Bones
            for start_point, end_point in SKELETON:

                if confidence[start_point] > 0.3 and confidence[end_point] > 0.3:

                    x1, y1 = points[start_point]
                    x2, y2 = points[end_point]

                    cv2.line(
                        canvas,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 255),
                        4
                    )

            # Draw Joints
            for index, (x, y) in enumerate(points):

                if confidence[index] > 0.3:

                    cv2.circle(
                        canvas,
                        (int(x), int(y)),
                        8,
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        canvas,
                        str(index),
                        (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )

    # Performance Information

    cv2.putText(
        canvas,
        f"FPS : {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        canvas,
        f"Inference : {inference:.1f} ms",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"YOLO : {speed['inference']:.1f} ms",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow(window, canvas)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()