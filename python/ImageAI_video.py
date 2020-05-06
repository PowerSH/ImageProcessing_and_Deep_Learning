# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:51:07 2020

@author: psh95
"""

from imageai.Detection import VideoObjectDetection
import os

detector = VideoObjectDetection()

path = os.getcwd() + "/data/"

detector.setModelTypeAsYOLOv3()
detector.setModelPath(path + "yolo.h5")
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=path + "sample_video2.mp4",
                                             output_file_path=path + "detected_video2",
                                             frames_per_second=20, log_progress=True)
# It takes long long time
print(video_path)