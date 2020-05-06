# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:54:18 2020

@author: psh95
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

path = os.getcwd() + '/data/'
origin = cv2.cvtColor(cv2.imread(path + "Lenna.jpg"), cv2.COLOR_BGR2RGB)

# modify "val"
val = 33

kernel = np.ones((5, 5), np.float32)/25
blur = cv2.filter2D(origin, -1, kernel)
averaging = cv2.blur(origin, (val, val))
gaussian = cv2.GaussianBlur(origin, (val, val), 0)
median = cv2.medianBlur(origin, val)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(origin)
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(blur)
ax2.set_title("Filter Blur")
ax2.axis("off")

ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(averaging)
ax3.set_title("Averaging Blur")
ax3.axis("off")

ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(gaussian)
ax4.set_title("Gaussian Blur")
ax4.axis("off")

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(median)
ax5.set_title("Median Blur")
ax5.axis("off")

print("Blurring")
plt.show()