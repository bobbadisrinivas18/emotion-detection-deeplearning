import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.utils import load_fer2013, preprocess_image, EMOTIONS

def build_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.3),
        Dense(num_classes,activation='softmax')
    ])
    return model

def main(args):
    os.makedirs("saved_models", exist_ok=True)

    print("Loading FER2013 data...")
    X_train, X_val, y_train, y_val = load_fer2013(args.fer_csv)

    model = build_model(num_classes=len(EMOTIONS))
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    checkpoint = ModelCheckpoint("saved_models/best_model.h5", save_best_only=True, monitor="val_accuracy")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=[checkpoint]
    )

    model.save("saved_models/final_model.h5")
    print("Training complete! Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fer_csv", type=str, default="data/fer2013.csv")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)
