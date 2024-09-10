import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw


def detect_faces(image: np.ndarray, combine_methods: bool = False):
    # Prepare the image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # If combined_methods is True, use both Haar Cascade and HOG
    if combine_methods:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        haar_faces = [(y, x + w, y + h, x) for (x, y, w, h) in haar_faces]

        hog_faces = face_recognition.face_locations(rgb_image, model="hog")

        combined_faces = list(haar_faces) + list(hog_faces)

        # Remove duplicates
        filtered_faces = []
        for face in combined_faces:
            if not any(is_close(face, f) for f in filtered_faces):
                filtered_faces.append(face)
    else:
        # Only use HOG method
        filtered_faces = face_recognition.face_locations(rgb_image, model="hog")

    # Draw rectangles on the original image for detected faces
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left) in filtered_faces:
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)

    del draw
    # pil_image.save('output_image.jpg')

    return filtered_faces


def is_close(face1, face2, threshold=30):
    top1, right1, bottom1, left1 = face1
    top2, right2, bottom2, left2 = face2
    return (abs(top1 - top2) < threshold and
            abs(right1 - right2) < threshold and
            abs(bottom1 - bottom2) < threshold and
            abs(left1 - left2) < threshold)
