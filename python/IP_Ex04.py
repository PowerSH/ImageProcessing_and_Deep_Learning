# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:55:06 2020

@author: psh95
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

path = os.getcwd() + '/data/'
val = 64

origin = cv2.cvtColor(cv2.imread(path + "Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
noise = cv2.cvtColor(cv2.imread(path + "Lenna_noise.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
denoise = cv2.fastNlMeansDenoisingColored(noise, None, 15, 15, 7, 21)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(origin)
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(noise)
ax2.set_title("Noise")
ax2.axis("off")

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(denoise)
ax3.set_title("Denoise")
ax3.axis("off")