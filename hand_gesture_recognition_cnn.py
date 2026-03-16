# -*- coding: utf-8 -*-
"""
Hand Gesture Recognition using CNN
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------------
# Dataset Path
# -------------------------------
dataset_path = "leapGestRecog"

data = []
labels = []

print("Loading Images...")

# -------------------------------
# Load Images Safely
# -------------------------------
for subject in os.listdir(dataset_path):

    subject_path = os.path.join(dataset_path, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):

        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for img in os.listdir(gesture_path):

            img_path = os.path.join(gesture_path, img)

            # Read only PNG images
            if not img_path.endswith(".png"):
                continue

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (64, 64))

            data.append(image)
            labels.append(gesture)

# -------------------------------
# Convert to numpy arrays
# -------------------------------
data = np.array(data)
labels = np.array(labels)

print("Total images loaded:", len(data))

# -------------------------------
# Normalize images
# -------------------------------
data = data / 255.0
data = data.reshape(-1, 64, 64, 1)

# -------------------------------
# Convert labels to numbers
# -------------------------------
label_names = np.unique(labels)

label_dict = {}
for i, name in enumerate(label_names):
    label_dict[name] = i

labels_num = np.array([label_dict[i] for i in labels])

# One-hot encoding
labels_cat = to_categorical(labels_num)

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_cat, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------
# CNN Model
# -------------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(label_names), activation='softmax'))

# -------------------------------
# Compile Model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    batch_size=32
)

# -------------------------------
# Evaluate Model
# -------------------------------
loss, acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", acc)

# -------------------------------
# Save Model (Updated Format)
# -------------------------------
model.save("gesture_model.keras")

print("Model saved as gesture_model.keras")

# -------------------------------
# Plot Accuracy Graph
# -------------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.show()