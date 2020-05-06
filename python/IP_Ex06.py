# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:56:23 2020

@author: psh95
"""

import cv2
from matplotlib import pyplot as plt
import os

path = os.getcwd() + '/data/'
origin = cv2.cvtColor(cv2.imread(path + "Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY), cv2.COLOR_BGR2RGB)

canny = cv2.cvtColor(cv2.Canny(origin, 127, 255), cv2.COLOR_BGR2RGB)
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(origin)
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(canny)
ax2.set_title("Canny")
ax2.axis("off")

ax3 = fig.add_subplot(2, 2, 3)
ax3.imshow(sobel)
ax3.set_title("Sobel")
ax3.axis("off")

ax4 = fig.add_subplot(2, 2, 4)
ax4.imshow(laplacian)
ax4.set_title("Laplacian")
ax4.axis("off")

print("Edge Detecting")
plt.show()