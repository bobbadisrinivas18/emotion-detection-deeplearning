import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import preprocess_image, EMOTIONS, decode

def run_demo(model_path, cascade_path=None, cam=0, img_size=48):
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    model = load_model(model_path)

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            img = preprocess_image(face, size=img_size)
            if img is None:
                continue
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            label, conf = decode(pred)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model (e.g. saved_models/best_model.h5)")
    parser.add_argument("--cascade", default=None, help="Optional path to haarcascade xml")
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    run_demo(args.model, args.cascade, args.cam)
