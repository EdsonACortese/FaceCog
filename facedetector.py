import cv2 as cv
cap = cv.VideoCapture(0)
def detect(image):
    img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    classifier = cv.CascadeClassifier()
    classifier.load(cv.samples.findFile('./data/haarcascade_frontalface.xml'))
    faces = classifier.detectMultiScale(img)
    for (x,y,w,h) in  faces:
        cv.imshow('face', image[y:y+h,x:x+w])

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detect(frame)
    if cv.waitKey(10) == 27:
        break