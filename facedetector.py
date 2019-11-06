import cv2 as cv
import numpy as np

def detect(image):
    classifier = cv.CascadeClassifier()
    classifier.load(cv.samples.findFile('./data/haarcascade_frontalface.xml'))
    faces = classifier.detectMultiScale(image)
    heights = [face[-1] for face in faces]
    f = [0,0,0,0]
    if(heights):
        m = max(heights)
        f = 0
    for i in range(len(faces)):
        if(faces[i][-1] == m):
            f = faces[i]
    if(f[2]==0 and f[3]==0):
        return -1,image
    else:
        return 1,image[f[1]:f[1]+f[3],f[0]:f[0]+f[2]]

if __name__ =='__main__':
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        print(detect(frame))
