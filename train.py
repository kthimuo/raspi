import sys,os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from scipy import ndimage
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

root_dir = "./data/"
categories = ['takenaka','tamura','doi']
nb_classes = len(categories)
image_size = 32

X = []
Y = []
for name in categories:
    cat = categories.index(name)
    path = root_dir+name +'/'
    images = os.listdir(path)
    for img in images:
        img_path = path + img
        x = cv2.imread(img_path)
        try : 
            x = cv2.resize(x,(100,100))
            x = x /255
            X.append(x)
            Y.append([cat])
        except :
            pass

#print(len(X))
#print(x.shape)
X = np.array(X)
Y = np.array(Y)
Y = np_utils.to_categorical(Y, nb_classes)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,shuffle=True)
#datagen = ImageDataGenerator(
#                    vertical_flip=True,
#                        rotation_range=90,
#                                height_shift_range=10,
#                                    horizontal_flip=True)
#datagen.fit(X_train)

import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import time

def get_model():
    input_tensor = Input(shape=(100,100, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    vgg_model = Model(input=vgg16.input, output=top_model(vgg16.output))

    for layer in vgg_model.layers[:15]:
        layer.trainable = False

    vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])
    return vgg_model
model = get_model()

batch_size = 64
epochs = 20
#model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
#                                epochs=20)
model.fit(X_train,y_train,batch_size=256,epochs=10)
print(model.evaluate(X_test, y_test))
model.save('model1.h5')
#model.predict(X_test[0])
