# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:52:26 2020

@author: psh95
"""

import cv2
from matplotlib import pyplot as plt
import os

path = os.getcwd() + '/data/'
origin = cv2.cvtColor(cv2.imread(path+"Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

processed = cv2.bitwise_not(origin)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(origin)
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(processed)
ax2.set_title("Processed")
ax2.axis("off")

print("Reverse Color")
plt.show()