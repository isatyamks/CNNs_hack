import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load train data
df_train = pd.read_csv("Dataset\\train.csv")
image_dir = "Dataset\\Images"
image_size = (48, 48)  # Resize images to 48x48

# Prepare training data
X_train, y_train = [], []
label_mapping = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5, "surprise": 6}

for _, row in df_train.iterrows():
    img_path = os.path.join(image_dir, row["Image_name"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is not None:
        img = cv2.resize(img, image_size)
        X_train.append(img)
        y_train.append(label_mapping[row["Emotion"]])

X_train = np.array(X_train) / 255.0  # Normalize
X_train = X_train.reshape(-1, 48, 48, 1)  # Reshape for CNN
y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)  # One-hot encode labels

# Define CNN Model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save("emotion_model.h5")
model.save("emotion_model.keras")
