import numpy as np
import cv2
import sys,os

target = 'doi'
os.makedirs('./data/' + target, exist_ok=True)
cap = cv2.VideoCapture(0)

face_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml'
#eye_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml'
#fullbody_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_fullbody.xml'
#
face_cascade = cv2.CascadeClassifier(face_cascade_path)
#eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
#fullbody_cascade = cv2.CascadeClassifier(fullbody_cascade_path)
cnt = 0
while(True):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
#    full_bodys = fullbody_cascade.detectMultiScale(frame_gray)

    if len(faces) == 1:
        for x, y, w, h in faces:
#            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y: y + h, x: x + w]
            path = './data/'+ target + '/' + str(cnt) +'.jpg'
            cv2.imwrite(path,face)
            cnt +=1
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q') or cnt==150:
        break
cap.release()
cv2.destroyAllWindow()
