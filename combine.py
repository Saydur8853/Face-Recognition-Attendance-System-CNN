import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Part 1: Capturing Faces ---
# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Setup webcam
cap = cv2.VideoCapture(0)

# Get the person's name
person_name = input("Enter the person's name: ")
save_path = f'faces/{person_name}'
os.makedirs(save_path, exist_ok=True)

# Capture images
count = 0
while count < 50:  # Adjust the number of images as needed
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))  # Resize to standard size
        cv2.imwrite(f"{save_path}/{count}.jpg", face_resized)
        count += 1

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {person_name}.")

# --- Part 2: Training the Model ---
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

# Train the model
model.fit(data, labels, epochs=10, validation_split=0.2)

# Save the trained model
model.save("face_recognition_model.h5")

# Save the label map
with open("label_map.txt", "w") as f:
    f.write(str(label_map))

print("Model trained and saved successfully!")
