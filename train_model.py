import cv2
import numpy as np
from PIL import Image
import os

# Trening uchun katalog (rasmlar saqlanadigan joy)
dataset_path = 'dataset'

# LBPH yuzni tanib olish modeli
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Haar kaskad modeli
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Tasvirlarni o'qib olish va ID'larni olish
def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Kulrang formatga o'tkazish
        img_numpy = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])  # ID'ni rasm nomidan olish
        faces = face_cascade.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids

# Yuzlarni o'rgatish
faces, ids = get_images_and_labels(dataset_path)
recognizer.train(faces, np.array(ids))

# Modelni saqlash
recognizer.write('face_trainer.yml')

print(f"[INFO] Model o'rgatildi va saqlandi.")
