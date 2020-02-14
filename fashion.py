from time import sleep
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from glob import glob
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS


Labels = []
classesFile1 = "yolo/obj.names";
with open(classesFile1, 'rt') as f:
    Labels = f.read().rstrip('\n').split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(Labels), 3), dtype="uint8")


weightsPath = "yolo/obj_4000.weights"
configPath = "yolo/obj.cfg"

net1 = cv.dnn.readNetFromDarknet(configPath, weightsPath)
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


image = WebcamVideoStream(src=0).start()
fps = FPS().start()
#'/home/raj/Documents/yolov3-Helmet-Detection-master/safety.mp4'

#while fps._numFrames<100:
while True:
#for fn in glob('images/*.jpg'):
    frame = image.read()
    #frame = imutils.resize(frame,width=500)
    (H, W) = frame.shape[:2]

    ln = net1.getLayerNames()
    ln = [ln[i[0] - 1] for i in net1.getUnconnectedOutLayers()]
    blob1 = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net1.setInput(blob1)
    layerOutputs = net1.forward(ln)

    boxes1 = []
    confidences1 = []
    classIDs1 = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes1.append([x, y, int(width), int(height)])
                confidences1.append(float(confidence))
                classIDs1.append(classID)

    idxs = cv.dnn.NMSBoxes(boxes1, confidences1, 0.5, 0.1)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes1[i][0], boxes1[i][1])
            (w, h) = (boxes1[i][2], boxes1[i][3])
            
            color = [int(c) for c in COLORS[classIDs1[i]]]
            
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            texts = "{}: {:.4f}".format(Labels[classIDs1[i]], confidences1[i])
            cv.putText(frame, texts, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)


    
    if 1>0:
        cv.imshow('img',frame)
        cv.waitKey(200) & 0xFF


    