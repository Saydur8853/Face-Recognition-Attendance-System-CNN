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
attendance_marked = {}

# Variable to store the attendance message
attendance_message = ""
message_time = None

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
                print(f"Attendance: {name}.")

                # Update the last attendance time
                last_attendance_time = now

                # Store the attendance message
                attendance_message = f"Attendance: {name} at {now.strftime('%Y-%m-%d %H:%M:%S')}"

                message_time = time.time()  # Record the time the message was shown

        # Display name on the screen above the face
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)


        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # If an attendance message exists and enough time has passed, display it on the screen
    if attendance_message and (time.time() - message_time) < 5:  # Show message for 5 seconds
        cv2.putText(frame, attendance_message, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Display the frame with annotated faces
    cv2.imshow("Face Recognition Attendance", frame)

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
