import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("emotion_model.keras")

# Load test data
df_test = pd.read_csv("Dataset\\test.csv")
image_dir = "Dataset\\Images"
image_size = (48, 48)

label_mapping = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sorrow": 5, "surprise": 6,"disregard": 7}

# Prepare test images
X_test, image_names = [], []

for _, row in df_test.iterrows():
    img_path = os.path.join(image_dir, row["Image_name"])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is not None:
        img = cv2.resize(img, image_size)
        X_test.append(img)
        image_names.append(row["Image_name"])

X_test = np.array(X_test) / 255.0  # Normalize
X_test = X_test.reshape(-1, 48, 48, 1)  # Reshape for CNN

# Make predictions
predictions = model.predict(X_test)
predicted_labels = [label_mapping[np.argmax(pred)] for pred in predictions]

# Save results to submission.csv
submission = pd.DataFrame({"Image_name": image_names, "Emotion": predicted_labels})
submission.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")
