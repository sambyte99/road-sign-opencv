import serial
import cv2
import time
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from audioplayer import AudioPlayer
import datetime

#firebass
from firebase import firebase
from tensorflow.python.saved_model import signature_def_utils  
firebase = firebase.FirebaseApplication('https://iot-rover-69-default-rtdb.firebaseio.com/', None)
COM="COM4"
BAUD=9600
SerialPort = serial.Serial(COM,BAUD,timeout=1)
OutgoingData ='z'
SerialPort.write(bytes(OutgoingData, 'utf-8'))

#configure gpu memory growth, in case of using a dedicated gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#image pre processing
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

#mouse click event handler function for main window
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

#import the sign descriptions as a list
data = pd.read_csv("signnames.csv")
df = data['SignName'].values.tolist()

#setup camera capture and display windows
cameraCapture = cv2.VideoCapture(1)
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse)

#read frame buffer from camera
success, frame = cameraCapture.read()

#load the saved model
loadedModel = keras.models.load_model('mymodel.hdf5')

#function to verify if the detected contour is a cicle, triangle, octagon or otherwise
def detectShape(c):
        peri=cv2.arcLength(maxContour,True)
        vertices = cv2.approxPolyDP(maxContour, 0.02 * peri, True)
        sides = len(vertices)
        if (sides == 3):
            return True
        elif(sides==8):
            return True
        return False

while success and not clicked:
    cv2.waitKey(1)
    success, frame = cameraCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the image to black and white
    edged = cv2.Canny(gray, 70, 255) #use Canny algorithm to detect edges

    (contours,_) = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find all possible contours

    maxContour = max(contours, key=cv2.contourArea) #select the biggest contour by area

    moment=cv2.moments(maxContour) #find the moments of the bigggest contour to find the centre
    if(moment['m00'] != 0):
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
    else:
        cx = 0
        cy = 0

    shape=detectShape(maxContour) #verify if the contour selected is among circle, triangle or octagon

    if shape and (cv2.contourArea(maxContour)>frame.shape[0]*frame.shape[1]*0.025): #proceed further only if the selected contour is minimus 2.5% of the camera resolution
        cv2.drawContours(frame,[maxContour],-1,(0,255,0),2) #draw the contour on the output window
        x,y,w,h = cv2.boundingRect(maxContour)
        square = frame[y-int(h*0.3):y+int(h*1.3), x-int(w*0.3):x+int(w*1.3)] #crop the image to 130% of the bounding rectangle of the contour

        if square.size != 0:
            resizedFrame = cv2.resize(square,(32,32)) #resize the frame to the size of the model, 32x32
            preProcessesFrame = preprocessing(resizedFrame)
            preProcessesFrame = preProcessesFrame.reshape(1,32,32,1) #reshape the pre processed frame to the shape required for the model

            signIndex = int(loadedModel.predict_classes(preProcessesFrame)) #precidt the class to which the sign belongs
            signDescription = df[signIndex] #retrieve the description of the sign

            print("Predicted sign: "+ signDescription)


            AudioPlayer(str(signIndex+1)+".mp3").play(block=True)
            ts=datetime.datetime.now()
            data={'Time':str(ts),'Sign':str(signDescription)}
            result = firebase.post('/Test/',data)  
            #print(result)
            if(signIndex<26):
                OutgoingData=chr(65+signIndex)
            else:
                OutgoingData=chr(97+signIndex-26)
            SerialPort.write(bytes(OutgoingData, 'utf-8'))


            cv2.putText(frame, text= signDescription, org=(cx,cy),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),
                thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('camera', frame)