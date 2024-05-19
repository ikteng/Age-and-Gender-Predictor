import pandas as pd
import numpy as np
import seaborn as sns
import os
from PIL import Image
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Evaluate the models
def process_and_predict(im):
    width, height = im.size
    if width == height:
        im = im.resize((200, 200), Image.LANCZOS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.LANCZOS)
        else:
            left = 0
            right = width
            top = height/2 - width/2
            bottom = height/2 + width/2
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.LANCZOS)
            
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 200, 200, 3)
    
    age = agemodel.predict(ar)
    gender = np.round(genmodel.predict(ar))
    
    if gender == 0:
        gender = 'male'
    elif gender == 1:
        gender = 'female'
        
    return int(age[0][0]), gender

# Load the models
agemodel = load_model(r'Age and Gender Prediction/age_model1.h5')
genmodel = load_model(r'Age and Gender Prediction/gender_model1.h5')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# OpenCV to capture live video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw the bounding box around each face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Convert the face region to a PIL Image
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Process and predict
        age, gender = process_and_predict(img)

        # Display the results on the frame
        cv2.putText(frame, f'Age: {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(frame, f'Gender: {gender}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Live Age and Gender Prediction', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
