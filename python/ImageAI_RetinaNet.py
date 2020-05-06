# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:49:37 2020

@author: psh95
"""

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.image import imread

from imageai.Detection import ObjectDetection

import os

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
path = os.getcwd() + "/data/"
detector.setModelPath(path + "resnet50_coco_best_v2.0.1.h5")
# https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5

detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=path + "input_image.jpg",

                                             output_image_path=path + "output_image.jpg")

for eachObject in detections:

    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

img = imread(path + 'output_image.jpg')

plt.imshow(img)