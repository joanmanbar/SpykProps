# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:27:46 2020

@author: jbarreto
"""

# poster_images









import time
start_time = time.time()

import os

import numpy as np

import matplotlib.pyplot as plt

from skimage import io
from skimage import feature
from skimage import filters
from skimage import color
from skimage import util
from skimage.measure import label, regionprops, perimeter
from skimage import morphology
from skimage.morphology import medial_axis, skeletonize
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import exposure
from skimage import segmentation
from skimage import img_as_float

from glob import glob

import imageio as iio

from skan import draw
from skan import skeleton_to_csgraph
from skan import Skeleton, summarize

import scipy
from scipy import ndimage as ndi

import pandas as pd




















img0 = iio.imread(r"J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\40709_D25_308.tif")
# img0 = iio.imread(r"J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\40305_Y17_292.tif")
# img0 = iio.imread(r"J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\41010_G27_300.tif")


# Save image
fig, ax = plt.subplots()
# Do the plot code
ax.imshow(img0)
plt.show()
fig.tight_layout()
# fig.savefig('myimage.svg', format='svg', dpi=1200)
plt.savefig(r'J:\My Drive\M.Sc\Events\CSSA\Poster\rgb_0.jpeg', format='jpeg', dpi=600)




# Crop image 
io.imshow(img0)
# img1 = img0[5850:6250, 1140:4400,:]
# img1 = img0[1000:4500, 3500:4600,:]
# img1 = img0[1850:2500, 1300:4200,:] # For image 292
img1 = img0[5400:6200, 450:4150,:] # For image 300
io.imshow(img1)
image_rescaled = rescale(img1, 0.5, preserve_range=True, multichannel=True)
img2 = image_rescaled.astype(np.uint8)
io.imshow(img2)

# ORIGINAL
bw0 = img1[:, :, 0] > 40
io.imshow(bw0)

bw1 = morphology.remove_small_objects(bw0, min_size=5000) # Filter out objects whose area < 5000
io.imshow(bw1)

# Apply mask to ORIGINAL RGB
img3 = np.asarray(img1)
img3 = np.where(bw1[...,None], img1, 0)
io.imshow(img3)


# RESCALED
bw0_r = img2[:, :, 0] > 40
io.imshow(bw0_r)

bw1_r = morphology.remove_small_objects(bw0_r, min_size=5000) # Filter out objects whose area < 10000
io.imshow(bw1_r)
# Apply mask to RESCALED RGB
img3 = np.asarray(img2)
img3 = np.where(bw1_r[...,None], img2, 0)
io.imshow(img3)


fig, ax = plt.subplots()
# Do the plot code
ax.imshow(img3)
plt.show()
fig.tight_layout()
# fig.savefig('myimage.svg', format='svg', dpi=1200)
plt.savefig(r'J:\My Drive\M.Sc\Events\CSSA\Poster\rgb_1.jpeg', format='jpeg', dpi=600)


# Label spikes
# myspk = bw1_r
myspk = bw1
skel0 = morphology.skeletonize(myspk)
io.imshow(skel0)

# dilate skeleton
se0 = morphology.disk(2) 
dilated_skel0 = morphology.dilation(skel0, selem=se0, out=None)


# Plot skeleton on binary
fig, ax = plt.subplots()
draw.overlay_skeleton_2d(myspk, skel0, dilate=2, axes=ax);   
fig.tight_layout()
plt.savefig(r'J:\My Drive\M.Sc\Events\CSSA\Poster\skel_0.jpeg', format='jpeg', dpi=600)



distance = ndi.distance_transform_edt(myspk)
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=50, labels=myspk)
stem = morphology.remove_small_objects(local_maxi, min_size=5)
new_local_max = img_as_float(local_maxi) - img_as_float(stem)
new_local_max = new_local_max.astype(np.bool)

markers = ndi.label(new_local_max)[0]
labeled_spikelets = segmentation.watershed(-distance, markers, mask=myspk)

# Plot watershed
fig, ax = plt.subplots()
plt.imshow(labeled_spikelets, cmap=plt.cm.nipy_spectral)
plt.savefig(r'J:\My Drive\M.Sc\Events\CSSA\Poster\spikelets_0.jpeg', format='jpeg', dpi=600)



# Plot 3 images
fig, axes = plt.subplots(nrows=3, figsize=(24, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img3, cmap=plt.cm.gray)
ax[0].set_title('(A) Isolated Spike', fontsize=20)
ax[1].imshow(dilated_skel0)
ax[1].set_title('(B) Branching Patterns', fontsize=20)
ax[2].imshow(labeled_spikelets, cmap=plt.cm.nipy_spectral)
ax[2].set_title('(C) Detected spikelets', fontsize=20)

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
plt.savefig(r'J:\My Drive\M.Sc\Events\CSSA\Poster\three_0.jpeg', format='jpeg', dpi=600)































