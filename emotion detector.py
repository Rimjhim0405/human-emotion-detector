import cv2 
from tensorflow import deeplake
ds = deeplake.load('hub://activeloop/cifar10-train')
import numpy as np

# Load the pre-trained Haar cascade file for face detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Load the pre-trained model for emotion detection
emotion_model_path = 'path_to_emotion_model.h5'
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_model = deeplake
ds = deeplake.load('hub://activeloop/fer2013-train')
dataloader = ds.tensorflow()

# Load the pre-trained Haar cascade file for human detection
cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# Open the webcam
video = cv2.VideoCapture(0)

# Continuously detect humans in the live video stream
while True:
    # Read the current frame from the video stream
    ret, frame = video.read()
    if not ret:
      print("Failed to read frame from the video stream")


    # Convert the frame to grayscale for human detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform human detection
    humans = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))

    # Draw rectangles around the detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame with human detections
    cv2.imshow('Human Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()