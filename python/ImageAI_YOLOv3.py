# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:41:09 2020

@author: psh95
"""

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.image import imread

from imageai.Detection import ObjectDetection

import os

detector = ObjectDetection()

# detector.setModelTypeAsRetinaNet()
detector.setModelTypeAsYOLOv3()
# detector.setModelTypeAsTinyYOLOv3()
path = os.getcwd() + "/data/"
detector.setModelPath(path + "yolo.h5")

detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=path + "input_image2.jpg",

                                             output_image_path=path + "output_image2.jpg")

for eachObject in detections:

    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

img = imread(path + 'output_image2.jpg')

plt.imshow(img)