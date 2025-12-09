import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = './data'
EXPECTED_FEATURES = 42  # 21 landmarks × 2 coords

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    max_num_hands=1
)

data = []
labels = []

def extract_features(landmarks):
    x = [lm.x for lm in landmarks.landmark]
    y = [lm.y for lm in landmarks.landmark]

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    width = max_x - min_x or 1
    height = max_y - min_y or 1

    feat = []
    for i in range(21):
        feat.append((x[i] - min_x) / width)
        feat.append((y[i] - min_y) / height)

    return feat

for folder in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(class_dir):
        continue

    print("Processing:", folder)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        features = extract_features(results.multi_hand_landmarks[0])

        if len(features) == EXPECTED_FEATURES:
            data.append(features)
            labels.append(folder)

print("Total samples:", len(data))

with open("data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Dataset saved → data.pickle")