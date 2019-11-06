import os

import cv2
import numpy as np
import tensorflow as tf

from facedetector import detect

def identify(image):
    ret, img = detect(image)
    if(ret == -1):
        return 0
    else:
        return isme(image)
def prepare(file):
    IMG_SIZE = 100
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
def isme(image):
    model = tf.keras.models.load_model("model.h5") 
    a = prepare(image)
    prediction = model.predict(a)
    prediction = prediction[0]
    return prediction[0]

