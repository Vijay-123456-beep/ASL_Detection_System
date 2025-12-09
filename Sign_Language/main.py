import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import time
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------
# Load SVM Model + Scaler
# ----------------------------------------------------------
model_data = pickle.load(open("model.p", "rb"))
model = model_data["model"]
scaler = model_data["scaler"]

EXPECTED_FEATURES = 42
CONFIDENCE_THRESHOLD = 0.55

# ----------------------------------------------------------
# Label Mapping (folder names → chars)
# ----------------------------------------------------------
def translate(label):
    if label == "space": return " "
    if label == "dot": return "."
    return label

# ----------------------------------------------------------
# Mediapipe Setup
# ----------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    max_num_hands=1
)

# ----------------------------------------------------------
# Speech
# ----------------------------------------------------------
engine = pyttsx3.init()

def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

# ----------------------------------------------------------
# Prediction Buffers (Speed + Stability)
# ----------------------------------------------------------
BUFFER_SIZE = 14
SMOOTH_COUNT = 9
registration_delay = 0.65
CONFIDENCE_THRESHOLD = 0.45


buffer = []
word = ""
sentence = ""
last_time = time.time()

# ----------------------------------------------------------
# GUI Setup
# ----------------------------------------------------------
root = tk.Tk()
root.title("American Sign Language Translator")
root.geometry("1400x800")
root.configure(bg="white")
root.resizable(False, False)

current_char = StringVar(value="---")
current_word = StringVar(value="---")
current_sentence = StringVar(value="---")

Label(root, text="American Sign Language Translator",
      font=("Helvetica", 32, "bold"), bg="white").pack(pady=20)

main_frame = Frame(root, bg="white")
main_frame.pack()

# Video Feed Widget
video_frame = Frame(main_frame, width=640, height=480)
video_frame.grid(row=0, column=0, padx=40)
video_label = Label(video_frame)
video_label.pack(expand=True, fill="both")

# Right Side UI
right_frame = Frame(main_frame, bg="white")
right_frame.grid(row=0, column=1, padx=40)

Label(right_frame, text="CURRENT SIGN", font=("Arial", 22, "bold"), bg="white").pack()
Label(right_frame, textvariable=current_char, font=("Arial", 48, "bold"), fg="green", bg="white").pack(pady=10)

Label(right_frame, text="CURRENT WORD", font=("Arial", 22, "bold"), bg="white").pack()
Label(right_frame, textvariable=current_word, font=("Arial", 20), fg="blue", bg="white").pack(pady=5)

Label(right_frame, text="SENTENCE", font=("Arial", 22, "bold"), bg="white").pack()
Label(right_frame, textvariable=current_sentence, wraplength=380, font=("Arial", 18), bg="white").pack(pady=5)

# Buttons
def reset_all():
    global word, sentence, buffer
    word = ""
    sentence = ""
    buffer = []
    current_char.set("---")
    current_word.set("---")
    current_sentence.set("---")

btn_frame = Frame(root, bg="white")
btn_frame.pack(pady=20)

Button(btn_frame, text="RESET", command=reset_all, fg="white", bg="red",
       font=("Arial",16,"bold"), width=12).grid(row=0, column=0, padx=10)

Button(btn_frame, text="SPEAK", command=lambda: speak(current_sentence.get()),
       fg="white", bg="green", font=("Arial",16,"bold"), width=12).grid(row=0, column=1, padx=10)

# ----------------------------------------------------------
# Feature Extraction (same as dataset)
# ----------------------------------------------------------
def extract_features(hand):
    x = [lm.x for lm in hand.landmark]
    y = [lm.y for lm in hand.landmark]

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    width = max_x - min_x or 1
    height = max_y - min_y or 1

    f = []
    for i in range(21):
        f.append((x[i] - min_x) / width)
        f.append((y[i] - min_y) / height)

    return np.array(f).reshape(1, -1)

# ----------------------------------------------------------
# Main Loop
# ----------------------------------------------------------
cap = cv2.VideoCapture(0)

def loop():
    global buffer, word, sentence, last_time

    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        features = extract_features(hand)
        features = scaler.transform(features)

        probs = model.predict_proba(features)[0]
        conf = max(probs)

        if conf >= CONFIDENCE_THRESHOLD:
            label = model.predict(features)[0]
            char = translate(label)

            buffer.append(char)
            if len(buffer) > BUFFER_SIZE:
                buffer.pop(0)

            if buffer.count(char) >= SMOOTH_COUNT:
                now = time.time()
                if now - last_time > registration_delay:
                    current_char.set(char)
                    last_time = now

                    if char == " ":
                        if word.strip():
                            sentence += word + " "
                            current_sentence.set(sentence)
                        word = ""
                        current_word.set("---")
                    elif char == ".":
                        if word.strip():
                            sentence += word + ". "
                            current_sentence.set(sentence)
                        word = ""
                        current_word.set("---")
                    else:
                        word += char
                        current_word.set(word)

        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    frame_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.configure(image=frame_tk)
    video_label.image = frame_tk

    root.after(10, loop)

loop()
root.mainloop()