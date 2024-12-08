import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset
data = []
labels = []
label_map = {}

base_path = 'faces'
for i, person in enumerate(os.listdir(base_path)):
    label_map[i] = person
    person_path = os.path.join(base_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (128, 128)) / 255.0
        data.append(img_resized)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10, validation_split=0.2)
model.save("face_recognition_model.h5")

# Save label map
with open("label_map.txt", "w") as f:
    f.write(str(label_map))

print("Model trained and saved successfully!")
