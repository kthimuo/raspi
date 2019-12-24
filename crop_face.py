import sys,os
import numpy as np
import cv2

crop_target = 'maki'
face_cascade_path = '/Users/tamurakazuki/.pyenv/versions/3.5.4/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml'
op = './original_data/' + crop_target +'/'
rp = './data/' + crop_target + '/'

all_pic_names = os.listdir('./original_data/' + crop_target + '/')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

cnt = 0
for pic_name in all_pic_names:
    try : 
        path = op + pic_name
        img = cv2.imread(path)

        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray)

        if len(faces) == 1:
            for x, y, w, h in faces:
                cnt +=1
                face = img[y: y + h, x: x + w]
                cv2.imwrite(rp + crop_target + '_'+ str(cnt)+'.jpg', face)
    except :
        pass
