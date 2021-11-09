import cv2

# loading some cascades. Already downloaded

# cascades for the eyes
face_casca = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# cascades for the face
eye_casca = cv2.CascadeClassifier('.//haarcascade_eye.xml')

# func to detect faces


def detect_my_face(gray, frame):
    # getting coordinates of the face
    faces = face_casca.detectMultiScale(gray, 1.3, 5)
