# DOWNLOAD THE DATASET
!pip install pandas numpy tensorflow opencv-python matplotlib

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Download and extract dataset (corrected)
!wget -q https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar.gz
!mkdir fer2013  # Create directory first
!tar -xzf fer2013.tar.gz -C fer2013/  # Extract into directory
!rm fer2013.tar.gz

# Verify the file exists
!ls fer2013/fer2013/  # Check contents

# Load the CSV file
df = pd.read_csv('fer2013/fer2013/fer2013.csv')  # Note the double fer2013/fer2013/
print(df.head())

#PREPROCESS THE DATA
# Convert pixel strings to numpy arrays
def parse_pixels(pixel_str):
    return np.array(pixel_str.split(), dtype='float32').reshape(48, 48, 1)

# Apply to all images
X = np.array([parse_pixels(pixels) for pixels in df['pixels']])
y = df['emotion'].values

# Normalize pixel values
X = X / 255.0

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

#VISULAIZE SAMPLE IMAGES
plt.figure(figsize=(12, 6))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.title(emotion_labels[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

#TRAIN MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, RandomFlip, RandomRotation
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data Augmentation Layer (applied during training only)
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
], name="data_augmentation")

# Improved Model Architecture
model = Sequential([
    data_augmentation,
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Custom Learning Rate Schedule
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=1000, decay_rate=0.9
)

# Compile with Adam optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Install required packages
!pip install opencv-python numpy tensorflow

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from base64 import b64decode
import time

# Load your trained model
model = load_model('emotion_model.h5')  # Make sure this matches your saved filename
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# JavaScript to create webcam stream in google colab
def start_webcam():
    js = Javascript('''
    async function start() {
        const video = document.createElement('video');
        const stream = await navigator.mediaDevices.getUserMedia({video: true});

        video.srcObject = stream;
        await video.play();

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');

        // Mirror the video
        context.translate(canvas.width, 0);
        context.scale(-1, 1);

        setInterval(() => {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imgData = canvas.toDataURL('image/jpeg', 0.8);
            google.colab.kernel.invokeFunction('notebook.processFrame', [imgData], {});
        }, 100);
    }
    ''')
    display(HTML("<div id='video-container'></div>"))
    display(js)
    eval_js('start()')

# Global variable to store frames
latest_frame = None

def process_frame(img_data):
    global latest_frame
    latest_frame = img_data

# Register callback
from google.colab import output
output.register_callback('notebook.processFrame', process_frame)

def base64_to_image(base64_data):
    image_bytes = b64decode(base64_data.split(',')[1])
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

# Start webcam
print("Starting webcam - please allow camera access when prompted")
start_webcam()

# Main processing loop
try:
    while True:
        if latest_frame:
            # Get and mirror frame
            frame = base64_to_image(latest_frame)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Preprocess face for emotion detection
                face_roi = gray[y:y+h, x:x+w]
                resized = cv2.resize(face_roi, (48, 48))
                normalized = resized / 255.0
                input_tensor = np.expand_dims(np.expand_dims(normalized, -1), 0)

                # Predict emotion
                preds = model.predict(input_tensor)
                emotion_idx = np.argmax(preds)
                emotion = emotion_labels[emotion_idx]
                confidence = np.max(preds)

                # Display emotion
                cv2.putText(frame, f"{emotion} ({confidence:.2f})",
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display in Colab
            _, buffer = cv2.imencode('.jpg', frame)
            display(Image(data=buffer.tobytes()))

            # Clear output between frames
            time.sleep(0.1)
            output.clear()

except KeyboardInterrupt:
    print("Webcam stopped")

