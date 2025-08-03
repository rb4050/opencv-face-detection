import cv2
import os
import numpy as np
from datetime import datetime

# Load DNN model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelFile = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.join(BASE_DIR, "deploy.prototxt.txt")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Create output directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(BASE_DIR, f"face_logs/session_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Saving unique detected faces to: {output_dir}")

# Initialize webcam
cap = cv2.VideoCapture(0)
saved_faces = []
face_id = 0

# Face comparison using histogram similarity
def is_new_face(new_face, saved_faces, threshold=0.6):
    new_hist = cv2.calcHist([new_face], [0], None, [256], [0, 256])
    new_hist = cv2.normalize(new_hist, new_hist).flatten()
    for saved in saved_faces:
        saved_hist = cv2.calcHist([saved], [0], None, [256], [0, 256])
        saved_hist = cv2.normalize(saved_hist, saved_hist).flatten()
        similarity = cv2.compareHist(new_hist, saved_hist, cv2.HISTCMP_CORREL)
        if similarity > threshold:
            return False
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Create input blob for DNN
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    confidence_threshold = 0.6  # Adjust if needed

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray_face_resized = cv2.resize(gray_face, (100, 100))

            if is_new_face(gray_face_resized, saved_faces):
                saved_faces.append(gray_face_resized)
                face_id += 1
                face_path = os.path.join(output_dir, f"face_{face_id}.jpg")
                cv2.imwrite(face_path, face)
                print(f"[INFO] Saved new face {face_id}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"New Face {face_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Duplicate", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("DNN Face Detection - Unique Faces Only", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Unique faces saved: {face_id}")
