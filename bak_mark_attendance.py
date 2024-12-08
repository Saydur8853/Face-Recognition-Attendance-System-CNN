import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import datetime
import os
import time

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

# Variable to store the last time attendance was marked
last_attendance_time = None
attendance_interval = 5  # Minimum time between entries in seconds

# Counter to track if attendance has been marked
attendance_marked = False

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

        # Get the current time
        now = datetime.datetime.now()

        # Check if enough time has passed since the last entry
        if last_attendance_time is None or (now - last_attendance_time).seconds >= attendance_interval:
            # Check if attendance for the person has already been marked today
            df = pd.read_excel(attendance_file)
            if not ((df["Name"] == name) & (df["Date"] == now.date())).any():
                new_entry = pd.DataFrame({"Name": [name], "Date": [now.date()], "Time": [now.time()]})
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_excel(attendance_file, index=False)
                print(f"Attendance marked for {name}.")

                # Update the last attendance time
                last_attendance_time = now

                # Mark the attendance as processed
                attendance_marked = True
                break  # Exit the loop after marking the attendance

    # Display name on the screen
    cv2.imshow("Face Recognition Attendance", frame)
    
    # Break the loop and release the camera if attendance is marked
    if attendance_marked:
        print("Attendance marked. Shutting down the camera...")
        break
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
