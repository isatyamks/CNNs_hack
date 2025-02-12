import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CSV
df_train = pd.read_csv("Dataset\\train.csv")
image_dir = "Dataset\\Images"
image_size = (224, 224)  # Ensure images are 224x224

X_train, y_train = [], []
label_mapping = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sorrow": 5, "surprise": 6, "disregard": 7}

# Read and process images
for _, row in df_train.iterrows():
    emotion = row["Emotion"]
    if emotion in label_mapping:
        img_path = os.path.join(image_dir, row["Image_name"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape != (224, 224):  # Resize only if necessary
                img = cv2.resize(img, image_size)
            X_train.append(img)
            y_train.append(label_mapping[emotion])

X_train = np.array(X_train) / 255.0  # Normalize images
X_train = X_train.reshape(-1, 224, 224, 1)  # Reshape for Conv2D
y_train = tf.keras.utils.to_categorical(y_train, num_classes=8)  # One-hot encode labels

# Data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')  # Output layer with 8 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=15, validation_split=0.2)

# Save the model
model.save("emotion_model.keras")
