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
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))
        roi_gray = gray[y:y+h, x:x+w]  # zone of interest in grayscale
        roi_col = frame[y:y+h, x:x+w]  # zone of interest in color
        eyes = eye_casca.detectMultiScale(
            roi_gray, 1.1, 5)  # detecting eyes using roi
        for (ex, ey, ew, eh) in eyes: #roi for eyes
            cv2.rectangle(roi_col, (ex, ey), (ex+ew, ey+eh), (0, 0, 255))
    return frame