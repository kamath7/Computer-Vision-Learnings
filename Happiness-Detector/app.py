#same code as previous
import cv2

# loading some cascades. Already downloaded

# cascades for the eyes
face_casca = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# cascades for the face
eye_casca = cv2.CascadeClassifier('.//haarcascade_eye.xml')
smile_casca = cv2.CascadeClassifier('./haarcascade_smile.xml')
# func to detect faces


def detect_my_face(gray, frame):
    # getting coordinates of the face
    faces = face_casca.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0))
        roi_gray = gray[y:y+h, x:x+w]  # zone of interest in grayscale
        roi_col = frame[y:y+h, x:x+w]  # zone of interest in color
        eyes = eye_casca.detectMultiScale(
            roi_gray, 1.1, 3)  # detecting eyes using roi
        for (ex, ey, ew, eh) in eyes:  # roi for eyes
            cv2.rectangle(roi_col, (ex, ey), (ex+ew, ey+eh), (0, 0, 255))
        smile = smile_casca.detectMultiScale(roi_gray, 1.1, 3)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_col, (sx, sy), (sx+sw, sy+sh), (0, 255, 0) )
    return frame


# running the webcam 😉
vid_capture = cv2.VideoCapture(0)  # 0 - integrated webcam else use 1
while True:  # continuous frame
    _, frame = vid_capture.read()  # gets frames from the webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting to grayscale
    canvas = detect_my_face(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_capture.release()
cv2.destroyAllWindows()
