import cv2
import time
import os

text = input("Enter text: ").upper()
for char in text:
    if char == " ":
        time.sleep(1)
        continue

    path = f"signs/{char}.jpg"

    if os.path.exists(path):
        img = cv2.imread(path)
        cv2.imshow("Sign", img)
        cv2.waitKey(800)  # delay between letters
    else:
        print(f"No sign for {char}")

cv2.destroyAllWindows()