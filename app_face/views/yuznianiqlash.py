
import cv2
import os
from django.shortcuts import render
from django.http import StreamingHttpResponse
from skimage.metrics import structural_similarity as ssim


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_dataset(folder='dataset'):
    faces = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, filename), 0)
            if img is not None:
                detected = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)
                if len(detected) > 0:
                    (x, y, w, h) = detected[0]
                    faces.append(img[y:y+h, x:x+w])
    return faces

dataset_faces = load_dataset()


def gen(camera):
    camera = cv2.VideoCapture(0)
    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )

        for (x, y, w, h) in faces:
            detected_face = gray[y:y+h, x:x+w]


            match_found = False
            for dataset_face in dataset_faces:
                resized = cv2.resize(detected_face, (dataset_face.shape[1], dataset_face.shape[0]))
                score, _ = ssim(dataset_face, resized, full=True)
                if score > 0.6:
                    match_found = True
                    break



            color = (0, 255, 0) if match_found  else (0, 0, 255)
            text = "O'xshash!" if match_found else "O'xshash emas!"


            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')

    camera.release()


def video_feed(request):
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed(request):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return StreamingHttpResponse("Kamera ochilmadi!", status=500)

    return StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')


def face(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')


def main(request):
    return render(request, 'main.html')







