import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Create data folder
DATA_PATH = "data"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Labels (you can change)
labels = ["A", "B"]

for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

cap = cv2.VideoCapture(0)

frame_count = 0
current_label = "A"   # change manually later

print("Press 's' to save data, 'q' to quit")

while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Save when 's' pressed
            key = cv2.waitKey(1)
            if key == ord('s'):
                np.save(os.path.join(DATA_PATH, current_label, str(frame_count)), landmarks)
                frame_count += 1
                print("Saved:", frame_count)

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()