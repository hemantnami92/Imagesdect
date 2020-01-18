import cv2,os
import numpy as np
from PIL import Image 

#path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.write(path+r'\trainer\trainer.yml')
recognizer.read('C:\\Users\\Gateway\\Desktop\\Images\\Face-Recognition-master\\trainer\\trainer.yml')
cascadePath = 'C:\\Users\\Gateway\\Desktop\\Images\\Face-Recognition-master\\Classifiers\\face.xml'
#cascadePath = path+'\Classifiers\face.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        if(nbr_predicted==5):
             nbr_predicted='santhu'
        elif(nbr_predicted==2):
             nbr_predicted='malli'
        elif(nbr_predicted==3):
             nbr_predicted='priyanka'
        elif(nbr_predicted==4):
             nbr_predicted='hema'
        elif(nbr_predicted==0):
            nbr_predicted='sai'
            
	
	
       
        cv2.putText(im,str(nbr_predicted)+"--"+str(conf), (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2) #Draw the text
        cv2.imshow('im',im)
        cv2.waitKey(10)
cv2.destroyAllWindows()







