import cv2
import mediapipe as mp
import numpy as np
import os

# Name of the gesture (change later for B, C, etc.)
gesture = "A"
data_path = os.path.join("../data", gesture)
os.makedirs(data_path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append([lm.x, lm.y, lm.z])
            np.save(os.path.join(data_path, f"{count}.npy"), np.array(landmark_list))
            count += 1
            print(f"Saved sample {count}")
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27 or count >= 100:  # Stops at 100 samples
        break

cap.release()
cv2.destroyAllWindows()