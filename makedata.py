import os
import random

import cv2
import numpy as np
import tensorflow as tf

import joblib
from facedetector import detect


def rotateImage(image, angle):
    row,col = image.shape
    center = tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def prep(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    retval, image = detect(image)
    image = cv2.resize(image, (100,100))
    i = [image.copy() for i in range(6)]
    for j in range(5):
        r = bool(random.getrandbits(1))
        if(r):
            i[j] = cv2.flip(i[j],1)
        i[j] = rotateImage(i[j],random.randrange(0,45,1))
    return i

X = []
y = []
categories = ['notme','me']

for cat in categories:
    path = os.path.join('data/images',cat)
    for img in os.listdir(path):
        try:
            p = prep(os.path.join(path,img))
            for pr in p:
                X.append(pr)
            for i in range(6):
                y.append(categories.index(cat))
        except:
            pass

X = np.array(X).reshape(-1,100,100,1)
y = np.array(y)
indices = np.arange(X.shape[0])

random.shuffle(indices)

X = X[indices]
y = y[indices]

with open('data/X.pickle', 'wb') as fil:
    joblib.dump(X, fil)
with open('data/y.pickle', 'wb') as fil:
    joblib.dump(y, fil)