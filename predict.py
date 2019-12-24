import numpy as np
import cv2
from keras.models import load_model

cap = cv2.VideoCapture(0)
model = load_model('model1.h5')

face_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eye_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml'
#fullbody_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_fullbody.xml'
#
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
#fullbody_cascade = cv2.CascadeClassifier(fullbody_cascade_path)
while(True):
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
#    full_bodys = fullbody_cascade.detectMultiScale(frame_gray)
#
    if len(faces) ==1:
        for x, y, w, h in faces:
            img = frame[y: y + h, x: x + w]
            img=cv2.resize(img, (100,100))#画像のshape
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            X=np.asarray(img)
            X=X.reshape(-1,100,100,3)#画像のshape
#            x=x/255
            pre=model.predict([X])[0]
            pre_val = np.max(pre)
            if pre_val > 0.95:
#                cv2.rectangle(frame, (10, 10), (100,100), (255, 0, 0), 2)
                pre = np.argmax(pre)
                if pre == 0:
                    print('takenaka' + ' : {}'.format(pre_val))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                elif pre ==1:
                    print('tamura' + ' : {}'.format(pre_val))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                elif pre==2:
                    print('doi' + ' : {}'.format(pre_val))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else :
                    print('iwasaki' + ' : {}'.format(pre_val))

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
