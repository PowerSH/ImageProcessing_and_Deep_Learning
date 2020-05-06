# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:56:49 2020

@author: psh95
"""

import cv2
from matplotlib import pyplot as plt
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
import os

path = os.getcwd() + '/data/'
origin = cv2.cvtColor(cv2.imread(path + "Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
detect = cv2.cvtColor(cv2.imread(path + "Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)


for (x,y,w,h) in faces:
  cv2.rectangle(detect, (x, y), (x+w, y+h), (255, 0, 0), 2)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(origin)
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(detect)
ax2.set_title("Detect")
ax2.axis("off")

print("Face Detecting")
plt.show()