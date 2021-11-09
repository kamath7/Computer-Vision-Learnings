import cv2

#loading some cascades. Already downloaded

#cascades for the eyes 
face_casca = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#cascades for the face
eye_casca = cv2.CascadeClassifier('.//haarcascade_eye.xml')