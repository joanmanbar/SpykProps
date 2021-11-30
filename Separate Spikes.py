
# Not currently working well (July 14 2021)


from CustomFunctions import *
import os
import numpy as np
import matplotlib.pyplot as plt
# from glob import glob
import pandas as pd
import glob
import skimage
import cv2


# Define image folder
mypath = r'./Images/ShatNurseTROE2020'
Images = glob.glob(mypath + '/**/*.tif', recursive=True)

# Open and image and remove background
img0_name = Images[2]
# plt.imshow(plt.imread(img0_name))
img0 = RemoveBackground(img0_name)


# Convert to gray
gray0 = img0 @ [0.2126, 0.7152, 0.0722]

# Get Lab values
# Lab = color.rgb2lab(img0)
    
# Threshold
otsu = filters.threshold_otsu(gray0)
bw0 = gray0 > otsu
bw1 = morphology.remove_small_objects(bw0, min_size=1.5e-05 * gray0.shape[0] * gray0.shape[1])

# Regionprops
labeled_spks, num_spikes = label(bw1, return_num = True)
props_spikes = regionprops(labeled_spks)

bw2 = labeled_spks == 1
# plt.imshow(bw2)
gray1 = gray0[500:1500, 1000:3500]
bw2 = bw2[500:1500, 1000:3500]
# plt.imshow(bw2)


gray1 = np.uint8(gray1)
contours, hier = cv2.findContours(gray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
M = cv2.moments(cnt)
print(M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

area = cv2.contourArea(cnt)

perimeter = cv2.arcLength(cnt,True)

k = cv2.isContourConvex(cnt)

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon, True)

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(gray1,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('img', gray1)
cv2.waitKey()

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(gray1[box],0,(0,0,255),2)
# plt.imshow(bw2)

cv2.imshow('img', gray1)
cv2.waitKey()
cv2.imwrite('testingbb.jpg', gray1)
















