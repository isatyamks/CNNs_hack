import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import UnidentifiedImageError 
# # Paths
# dataset_path = "Dataset"
# image_folder = os.path.join(dataset_path, "Images")
# train_csv = os.path.join(dataset_path, "train.csv")
# test_csv = os.path.join(dataset_path, "test.csv")

# # Load train data
# train_df = pd.read_csv(train_csv)
# test_df = pd.read_csv(test_csv)

# # Load images and labels
# img_size = (224, 224)
# x, y = [], []
# for _, row in train_df.iterrows():
#     img_path = os.path.join(image_folder, f"{row['Image_name']}")
#     img = load_img(img_path, target_size=img_size)
#     img = img_to_array(img)
#     img = preprocess_input(img)
#     x.append(img)
#     y.append(row['Emotion'])

# x = np.array(x)

# # Encode labels
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)
# y = to_categorical(y)


# dataset_path = "Dataset"
# image_folder = os.path.join(dataset_path, "Images")
# train_csv = os.path.join(dataset_path, "train.csv")

# # Load train data
# train_df = pd.read_csv(train_csv)

# # Image processing parameters
# img_size = (224, 224)

# # Lists to store processed data
# x = []
# y = []
# # Process images
# for _, row in train_df.iterrows():
#     img_filename = row['Image_name']  # No need to add .png
#     img_path = os.path.join(image_folder, img_filename)

#     if not os.path.exists(img_path):
#         print(f"Warning: {img_filename} not found. Skipping...")
#         continue

#     img = load_img(img_path, target_size=img_size)
#     img = img_to_array(img)
#     img = preprocess_input(img)

#     x.append(img)
#     y.append(row['Emotion'])

# # Convert to NumPy arrays
# x = np.array(x)
# y = np.array(y)







img_size = (224, 224)

# Initialize lists
x = []
y = []

# Process images
for _, row in train_df.iterrows():
    img_filename = str(row['image_num']).strip()  # Ensure it's a string and remove whitespace
    img_path = os.path.join(image_folder, img_filename)

    if not os.path.exists(img_path):
        print(f"❌ Warning: {img_filename} not found. Skipping...")
        continue

    try:
        # Open and preprocess image
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        img = preprocess_input(img)

        x.append(img)
        y.append(row['emotion'])

    except UnidentifiedImageError:
        print(f"❌ Warning: {img_filename} is corrupted or not a valid image. Skipping...")

# Convert lists to NumPy arrays (only if data is not empty)
if x and y:
    x = np.array(x)
    y = np.array(y)
    print(f"✅ Successfully loaded {len(x)} images.")
else:
    print("⚠️ No valid images were found. Please check your dataset.")









# Train-test split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Build model using ResNet50 as a feature extractor
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Load test images for prediction
x_test = []
test_image_nums = []
for _, row in test_df.iterrows():
    img_path = os.path.join(image_folder, f"{row['image_num']}.png")
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img)
    img = preprocess_input(img)
    x_test.append(img)
    test_image_nums.append(row['image_num'])

x_test = np.array(x_test)

# Predict emotions
y_pred = model.predict(x_test)
y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))

# Save predictions
output_df = pd.DataFrame({'image_num': test_image_nums, 'predicted_emotion': y_pred_labels})
output_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")
