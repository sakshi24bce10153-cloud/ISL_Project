import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib  # to save the model

# List all gestures you collected
gestures = ["A", "B", "C"]

X, y = [], []

# Load all gesture data
for idx, gesture in enumerate(gestures):
    folder = f"../data/{gesture}"  # path to gesture folder
    for file in os.listdir(folder):
        data = np.load(os.path.join(folder, file))
        X.append(data.flatten())  # convert 21x3 points into 63 features
        y.append(idx)            # label: 0 for A, 1 for B, 2 for C

X = np.array(X)
y = np.array(y)

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"Model trained! Accuracy: {accuracy*100:.2f}%")

# Save the trained model
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/hand_model.pkl")
print("Model saved in models/hand_model.pkl")