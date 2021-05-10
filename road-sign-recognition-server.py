import cv2
import time
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

data = pd.read_csv("signnames.csv")
df = data['SignName'].values.tolist()
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('camera')
cv2.namedWindow('cameraCropped')
cv2.setMouseCallback('camera', onMouse)


success, frame = cameraCapture.read()

loadedModel = keras.models.load_model('mymodel.hdf5')

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 70, 255)
    ret,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
    (contours,_) = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    maxContour = max(contours, key=cv2.contourArea)
    moment=cv2.moments(maxContour)
    if(moment['m00'] != 0):
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
    else:
        cx = 0
        cy = 0
    shape=detectShape(maxContour)
    if shape:
        cv2.drawContours(frame,[maxContour],-1,(0,255,0),2)
        cv2.putText(frame,"Found",(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)  #Putting name of polygon along with the shape
        x,y,w,h = cv2.boundingRect(maxContour)
        square = frame[y-int(h*0.3):y+int(h*1.3), x-int(w*0.3):x+int(w*1.3)]
        if square.size != 0:
            resizedFrame = cv2.resize(square,(32,32))
            preProcessesFrame = preprocessing(resizedFrame)
            preProcessesFrame = preProcessesFrame.reshape(32,32,1)
            print(loadedModel.predict(preProcessesFrame.reshape(1,32,32,1)))
            signIndex = int(
                    loadedModel.predict_classes(preProcessesFrame.reshape(1, 32, 32, 1)))
            signDescription = df[signIndex]
            print("Predicted sign: "+ signDescription)
            cv2.imshow('cameraCropped', square)
            cv2.putText(frame, text= signDescription, org=(cx,cy),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),
                thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('camera', frame)