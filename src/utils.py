import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def preprocess_image(img, size=48):
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size))
    img = img.astype("float32")/255.0
    return np.expand_dims(img, -1)

def load_fer2013(csv_path, size=48, split=0.1):
    import pandas as pd
    df = pd.read_csv(csv_path)

    X, y = [], []
    for _, row in df.iterrows():
        pixels = np.fromstring(row["pixels"], sep=' ')
        if pixels.size != size * size:
            continue
        img = pixels.reshape(size, size) / 255.0
        img = np.expand_dims(img, -1)
        X.append(img)
        y.append(row["emotion"])

    X = np.array(X)
    y = to_categorical(y, num_classes=len(EMOTIONS))
    return train_test_split(X, y, test_size=split, random_state=42)

def decode(pred):
    idx = int(np.argmax(pred))
    return EMOTIONS[idx], float(np.max(pred))
