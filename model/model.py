import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image, ImageFile
import os

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dir = 'Dataset\\Images/'
IMAGE_EXTENSION = '.png' 

train_df = pd.read_csv('Dataset\\train.csv')
test_df = pd.read_csv('Dataset\\test.csv')

# Check if 'Emotion' column exists
if 'Emotion' not in train_df.columns:
    raise KeyError("'Emotion' column not found in train.csv")
if 'Image_name' not in test_df.columns:
    raise KeyError("'Image_name' column not found in test.csv")

train_df = train_df[train_df['Image_name'].apply(lambda x: os.path.isfile(os.path.join(image_dir, x + IMAGE_EXTENSION)))]
test_df = test_df[test_df['Image_name'].apply(lambda x: os.path.isfile(os.path.join(image_dir, x + IMAGE_EXTENSION)))]

train_df['image_path'] = image_dir + train_df['Image_name'] + IMAGE_EXTENSION
test_df['image_path'] = image_dir + test_df['Image_name'] + IMAGE_EXTENSION

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = train_df['Emotion'].nunique()

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training and validation generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='Emotion',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='Emotion',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load EfficientNetB0 base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train top layers
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Fine-tune some base layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Prepare test generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

# Generate predictions
predictions = model.predict(test_generator)
predicted_indices = np.argmax(predictions, axis=1)

# Map indices to labels
class_labels = list(train_generator.class_indices.keys())
predicted_Emotions = [class_labels[idx] for idx in predicted_indices]

# Create submission file
submission = pd.DataFrame({
    'Image_name': test_df['Image_name'],
    'Emotion': predicted_Emotions
})
submission.to_csv('submission.csv', index=False)
