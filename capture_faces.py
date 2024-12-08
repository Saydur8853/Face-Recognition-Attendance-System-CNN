import cv2
import os

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