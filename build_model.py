import pandas as pd
import numpy as np
import seaborn as sns
import os
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Initialize lists
images = []
ages = []
genders = []

# Define the directory
image_dir = r'age and gender prediction/crop_part1'

# Load images and extract age and gender information
for i in os.listdir(image_dir)[0:8000]:
    split = i.split('_')
    ages.append(int(split[0]))
    genders.append(int(split[1]))
    images.append(Image.open(os.path.join(image_dir, i)))

# Create a DataFrame
images = pd.Series(list(images), name='Images')
ages = pd.Series(list(ages), name='Ages')
genders = pd.Series(list(genders), name='Genders')
df = pd.concat([images, ages, genders], axis=1)

# Balance the data by undersampling ages <= 4
under4s = df[df['Ages'] <= 4]
under4s = under4s.sample(frac=0.3, random_state=42)
df = df[df['Ages'] > 4]
df = pd.concat([df, under4s], ignore_index=True)

# Filter ages less than 80
df = df[df['Ages'] < 80]

# Filter out invalid gender values
df = df[df['Genders'] != 3]

# Prepare image data for model training
x = []
y_age = []
y_gender = []

for i in range(len(df)):
    img = df['Images'].iloc[i].resize((200, 200), Image.LANCZOS)
    ar = np.asarray(img)
    x.append(ar)
    y_age.append(int(df['Ages'].iloc[i]))
    y_gender.append(int(df['Genders'].iloc[i]))

x = np.array(x)
y_age = np.array(y_age)
y_gender = np.array(y_gender)

# Train test split
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(x, y_age, test_size=0.2, stratify=y_age, random_state=42)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(x, y_gender, test_size=0.2, stratify=y_gender, random_state=42)

# Build age prediction model
agemodel = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='relu')
])

agemodel.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.0001))

# Build gender prediction model
genmodel = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

genmodel.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_age = ModelCheckpoint('age and gender prediction/age_model_best.h5', monitor='val_loss', save_best_only=True)
checkpoint_gender = ModelCheckpoint('age and gender prediction/gender_model_best.h5', monitor='val_accuracy', save_best_only=True)

# Train the age model
train1 = datagen.flow(x_train_age, y_train_age, batch_size=32)
test1 = test_datagen.flow(x_test_age, y_test_age, batch_size=32)

history1 = agemodel.fit(train1, epochs=30, shuffle=True, validation_data=test1, callbacks=[early_stopping, checkpoint_age])

# Save the age model
agemodel.save('age and gender prediction/age_model.h5')

# Train the gender model
train2 = datagen.flow(x_train_gender, y_train_gender, batch_size=64)
test2 = test_datagen.flow(x_test_gender, y_test_gender, batch_size=64)

history2 = genmodel.fit(train2, epochs=30, shuffle=True, validation_data=test2, callbacks=[early_stopping, checkpoint_gender])

# Save the gender model
genmodel.save('age and gender prediction/gender_model.h5')
