import cv2
import numpy as np
import time

# -------------------- CONFIG --------------------
prototxt_path = 'models/MobileNetSSD_deploy (1).prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2
FRAME_SKIP = 2   # higher = more FPS, lower accuracy

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

np.random.seed(543210)
colors = np.random.uniform(0, 255, (len(classes), 3))

# -------------------- LOAD MODEL --------------------
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# -------------------- CAMERA SETUP --------------------
cap = cv2.VideoCapture(0)

# ✅ Handle camera failure
if not cap.isOpened():
    print("❌ Error: Camera not accessible")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------------------- VIDEO WRITER --------------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

# -------------------- FPS COUNTER --------------------
fps = 0
frame_count = 0
start_time = time.time()

# -------------------- MAIN LOOP --------------------
while True:
    ret, image = cap.read()
    if not ret:
        break

    frame_count += 1

    # ✅ Improve FPS (skip frames)
    if frame_count % FRAME_SKIP != 0:
        out.write(image)
        cv2.imshow("Detected Objects", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007,
        (300, 300),
        130
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            class_index = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype("int")

            label = f"{classes[class_index]}: {confidence * 100:.1f}%"

            cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_index], 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10 if y1 > 20 else y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[class_index],
                2
            )

    # -------------------- FPS DISPLAY --------------------
    fps += 1
    elapsed = time.time() - start_time
    if elapsed >= 1:
        display_fps = fps
        fps = 0
        start_time = time.time()

    cv2.putText(
        image,
        f"FPS: {display_fps}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # -------------------- SHOW & SAVE --------------------
    out.write(image)
    cv2.imshow("Detected Objects", image)

    # ✅ Quit button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- CLEANUP --------------------
cap.release()
out.release()
cv2.destroyAllWindows()
