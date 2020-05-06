# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:53:06 2020

@author: psh95
"""

import cv2
from matplotlib import pyplot as plt
import os

path = os.getcwd() + '/data/'
origin = cv2.cvtColor(cv2.imread(path+"Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY), cv2.COLOR_BGR2RGB)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(origin)
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(gray)
ax2.set_title("Gray")
ax2.axis("off")

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(binary)
ax3.set_title("Binary")
ax3.axis("off")

print("Binarization")
plt.show()