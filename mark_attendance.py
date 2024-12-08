import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import datetime
import os

# Load Haar cascade and model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("face_recognition_model.h5")

# Load label map
with open("label_map.txt", "r") as f:
    label_map = eval(f.read())

# Attendance file
attendance_file = "attendance.xlsx"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_excel(attendance_file, index=False)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        predictions = model.predict(face_resized)
        predicted_label = np.argmax(predictions)
        name = label_map[predicted_label]

        # Mark attendance
        now = datetime.datetime.now()
        df = pd.read_excel(attendance_file)
        if not ((df["Name"] == name) & (df["Date"] == now.date())).any():
            new_entry = pd.DataFrame({"Name": [name], "Date": [now.date()], "Time": [now.time()]})
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_excel(attendance_file, index=False)
            print(f"Attendance marked for {name}.")

        # Display name
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
