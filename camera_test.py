import numpy as np
import cv2

cap = cv2.VideoCapture(0)

#face_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml'
#eye_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml'
#fullbody_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_fullbody.xml'
#
#face_cascade = cv2.CascadeClassifier(face_cascade_path)
#eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
#fullbody_cascade = cv2.CascadeClassifier(fullbody_cascade_path)
while(True):
    ret, frame = cap.read()
#    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    faces = face_cascade.detectMultiScale(frame_gray)
#    full_bodys = fullbody_cascade.detectMultiScale(frame_gray)
#
#    for x, y, w, h in faces:
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#        face = frame[y: y + h, x: x + w]
#        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#        eyes = eye_cascade.detectMultiScale(face_gray)
#        print(len(eyes))
#        for (ex, ey, ew, eh) in eyes:
#            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#    for x, y, w, h in full_bodys:
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#
#
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
    if key == ord('s'):
        path = 'photo.jpg'
        cv2.imwrite(path,frame)
cap.release()
cv2.destroyAllWindow()
