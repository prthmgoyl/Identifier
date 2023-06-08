from sklearn.neighbors import KNeighborsClassifier


import cv2
import pickle
import numpy as np
import os


video = cv2.VideoCapture(0)
detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('data/names.pkl','rb') as f:
     LABELS = pickle.load(f)
with open('data/faces.pkl','rb') as f:
     FACES = pickle.load(f)

model = KNeighborsClassifier(n_neighbors=5)
model.fit( FACES , LABELS)    


while True:
    ret, frame = video.read()
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
       crop = frame[y:y+h , x:x+w , :]
       resize = cv2.resize(crop,(50,50)).flatten().reshape(1,-1)
       output =model.predict(resize)
       cv2.putText(frame, str(output[0]) ,(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
       cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key==ord('c'):
      break
 
video.release()
cv2.destroyAllWindows()


