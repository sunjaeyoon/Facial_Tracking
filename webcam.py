# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 01:19:52 2019

@author: Tim
"""
import numpy as np
import cv2
#import matplotlib.pyplot as plt

def detect_faces(f_cascade, colored_img, glasses = False, scaleFactor = 1.1):
    specs = cv2.imread('thuglifespecs.png', -1)
    #plt.imshow(specs)
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor = scaleFactor, minNeighbors =5)
    #print(faces)
    if not glasses:
        for (x,y,w,h) in faces:
            cv2.rectangle(img_copy, (x,y), (x+w,y+w),(125, 155, 155), 2)
            
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)
    if len(faces) > 0 and glasses:
        for (x,y,w,h) in faces:
            specs = cv2.resize(specs, (w, h//2))
            w, h, c = specs.shape
            for i in range(0, w):
                for j in range(0, h):
                    if specs[i, j][3] != 0:
                        img_copy[y + i + 15, x + j] = specs[i, j]
            
    return img_copy

def liveStream(cam, f_cascade):
    while(True):#Infinite Loop
        # Capture frame-by-frame
        _ , frame = cam.read() # ret, frame
        frame = cv2.resize(frame,(640,480))#(500,500))

        # Our operations on the frame come here
        frame = cv2.resize(frame,(640,480))#(500,500))
        frame = np.fliplr(frame)
        #segment = detect_faces(f_cascade,frame)
        segment = detect_faces(f_cascade,frame, True)
        # Display the resulting frame
        cv2.imshow('Segmented', segment)
        #cv2.imshow('Original', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()
    
def main():
    # Image Detector
    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    #Initialize Camera
    cap = cv2.VideoCapture(0)
    liveStream(cap, haar_face_cascade)
        
if __name__ == "__main__":
    main()
    
