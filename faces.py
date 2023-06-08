import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
data = []
name = input("Enter Your Name:")
i = 0
while True:
    ret, frame = video.read()
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
       crop = frame[y:y+h , x:x+w , :]
       resize = cv2.resize(crop,(50,50))
       if len(data)<=100 and i%10 == 0:
         data.append(resize)
       i = i+1
       cv2.putText(frame, str(len(data)),(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)  
       cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key==ord('c') or len(data)==100:
      break
 
video.release()
cv2.destroyAllWindows()


data = np.asarray(data)
data = data.reshape(100,-1)

if 'names.pkl' not in os.listdir('data/'):
   names = [name]*100
   with open('data/names.pkl','wb') as f:
      pickle.dump(names, f)

else:
   with open('data/names.pkl','rb') as f:
     names = pickle.load(f)
   names = names + [name]*100
   with open('data/names.pkl','wb') as f:
      pickle.dump(names, f)

if 'faces.pkl' not in os.listdir('data/'):
   with open('data/faces.pkl','wb') as f:
      pickle.dump(data, f)

else:
   with open('data/faces.pkl','rb') as f:
     olddata = pickle.load(f)
   olddata = np.append(olddata,data, axis=0)
   with open('data/faces.pkl','wb') as f:
      pickle.dump(olddata, f)     