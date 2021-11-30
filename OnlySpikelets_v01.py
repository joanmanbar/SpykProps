#!/usr/bin/env python3

# Code to detect number of spikelets (ONLY)




from CustomFunctions import *

import time
start_time = time.time()

import os

# This is for math
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
from skimage.transform import rescale, resize, downscale_local_mean, rotate
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

# import cv2

import math


import cv2

from skimage.filters import try_all_threshold

from skimage.filters import threshold_otsu







# # Gather the image files (change path)
Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\Images\IN_DIR\*.tif')

img0 = RemoveBackground(Images[0], Thres_chan=0, Thres_val=50)
plt.imshow(img0)


# Convert to gray
gray0 = img0 @ [0.2126, 0.7152, 0.0722]

# Threshold
otsu = filters.threshold_otsu(gray0)
bw0 = gray0 > otsu
bw1 = morphology.remove_small_objects(bw0, min_size=500)
# plt.imshow(bw1, cmap='gray')
bw1_small = rescale(bw1[...], 0.25, preserve_range=True, multichannel=False)
# image_rescaled = image_rescaled.astype(np.uint8)
plt.imshow(image_rescaled, cmap='gray')

myspk = bw1

distance = ndi.distance_transform_edt(myspk)
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=50, labels=myspk)
stem = morphology.remove_small_objects(local_maxi, min_size=5)
new_local_max = img_as_float(local_maxi) - img_as_float(stem)
new_local_max = new_local_max.astype(np.bool)

markers = ndi.label(new_local_max)[0]
labeled_spikelets = segmentation.watershed(-distance, markers, mask=myspk)
plt.imshow(labeled_spikelets)







# Edge detector
edge_frangi = filters.frangi(gray0, black_ridges=False)

# Resize
image_rescaled = rescale(edge_frangi[...], 0.25, preserve_range=True, multichannel=False)
# image_rescaled = image_rescaled.astype(np.uint8)
plt.imshow(image_rescaled, cmap='gray')


# Skeletonize
skel0 = morphology.skeletonize(image_rescaled > 0.2)
io.imshow(skel0)

# Closing
closed = morphology.closing(skel0)
plt.imshow(closed, cmap='gray')

# Skeletonize
skel1 = morphology.medial_axis(closed)
io.imshow(skel1)


fig, ax = plt.subplots()
draw.overlay_skeleton_2d(bw1_small, skel1, dilate=0, axes=ax);  














skel0 = morphology.skeletonize(edge_frangi)
io.imshow(skel0)



skel1 = scipy.ndimage.morphology.binary_fill_holes(edge_frangi)
io.imshow(skel1)
# plt.imshow(1-edge_frangi, cmap='gray')
edge_frangi = 1- edge_frangi
plt.imshow(edge_frangi, cmap='gray')
plt.imshow(1- edge_frangi, cmap='gray')
plt.imshow(edge_frangi<0.2, cmap='gray')

image_rescaled = rescale(edge_frangi[...], 0.25, preserve_range=True, multichannel=False)
# image_rescaled = image_rescaled.astype(np.uint8)
plt.imshow(image_rescaled>.3, cmap='gray')

contour = image_rescaled > 0.3
testing = 


w_tophat = white_tophat(image_rescaled)
plot_comparison(image_rescaled, w_tophat, 'white tophat')

dilated = dilation(w_tophat, morphology.disk(2) )
plt.imshow(dilated, cmap='gray')




bw2 = morphology.remove_small_objects(edge_frangi<0.5, min_size=50000)
plt.imshow(~bw2, cmap='gray')

skel0 = morphology.skeletonize(~bw2)
io.imshow(skel0)


eroded = erosion(image_rescaled)
plot_comparison(image_rescaled, eroded, 'white tophat')

eroded = erosion(eroded)
plt.imshow(eroded, cmap='gray')

skel0 = morphology.skeletonize(eroded)
io.imshow(skel0)



bw2 = morphology.remove_small_objects(closed>0.5, min_size=50000)
plt.imshow(~bw2, cmap='gray')

skel0 = morphology.skeletonize(~bw2)
io.imshow(skel0)

skel1 = morphology.remove_small_objects(skel0, min_size=10)
plt.imshow(skel1, cmap='gray')



closed = erosion(closed)
plt.imshow(closed, cmap='gray')




# Dilation
dilated = morphology.closing(bw1)
io.imshow(dilated)

# Blurry with Gaussian filter
blurry = filters.gaussian(dilated, sigma = 10)
io.imshow(blurry, cmap='gray')
myspk0 = blurry > 0.5
# io.imshow(myspk0)






# Regionprops
labeled_spks, num_spikes = label(myspk0, return_num = True)
props = regionprops(labeled_spks)


# watershed
from skimage.segmentation import watershed

myspk_rot = myspk

distance = ndi.distance_transform_edt(myspk0)
# io.imshow(distance)
# local_maxi = feature.peak_local_max(distance, indices=False, footprint=morphology.diamond(30), labels=myspk_rot)
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=50, labels=myspk0)
stem = morphology.remove_small_objects(local_maxi, min_size=5)
# io.imshow(img_as_float(local_maxi) - img_as_float(stem))

new_local_max = img_as_float(local_maxi) - img_as_float(stem)
new_local_max = new_local_max.astype(np.bool)

# local_maxi = feature.corner_peaks(distance, indices=False, min_distance=20, labels=myspk_rot)
# io.imshow(new_local_max)



markers = ndi.label(local_maxi)[0]
labeled_spikelets = segmentation.watershed(-distance, markers, mask=myspk0)
plt.imshow(labeled_spikelets)

regions_spikelets = regionprops(labeled_spikelets)

# n_Spikelets = int(labeled_spikelets[:,:].max())

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(myspk0, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labeled_spikelets, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()




# for spk in props:
#     spk = 3
#     myspk = labeled_spks == spk











# Edge detector
edge_frangi = filters.frangi(gray0)
plt.imshow(255-edge_frangi, cmap='gray')




image_rescaled = rescale(gray0[...], 0.25, preserve_range=True, multichannel=False)
final_image = image_rescaled.astype(np.uint8)
gray1 = final_image
plt.imshow(gray1)

result = ndimage.median_filter(gray0, size=5)
plot_comparison(gray1, result, "something")

from scipy.ndimage import gaussian_filter
result2 = gaussian_filter(gray1, sigma=3)
# closed = closing(edges3)
plot_comparison(result2, result, 'closing')


from PIL import Image, ImageFilter
from matplotlib import cm
gray_pil = Image.fromarray(np.uint8(cm.gist_earth(result2)*255))
edges_pil = gray_pil.convert("L")
edges_pil = edges_pil.filter(ImageFilter.FIND_EDGES)

edges_pil = np.array(edges_pil)
plt.imshow(edges_pil, cmap='gray')




edge_scharr = filters.scharr(gray0)
plt.imshow(edge_scharr>90, cmap='gray')



image3 = np.where(edge_scharr[..., None], image1, 0)        # third condition changes mask's background on a 1-255 scale
io.imshow(image3)  



from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt



edge_meijering = filters.meijering(gray0)
plt.imshow(edge_meijering, cmap='gray')

edge_sato = filters.sato(gray0)
plt.imshow(edge_sato, cmap='gray')

edge_frangi = filters.frangi(gray0)
plt.imshow(edge_frangi, cmap='gray')

edge_hessian = filters.hessian(gray0)
plt.imshow(edge_hessian, cmap='gray')


ComparePlots(1, 4, [edge_meijering, edge_sato, edge_frangi, edge_hessian])




image_rescaled = rescale(edge_frangi[...]*255, 0.25, preserve_range=True, multichannel=False)
image_rescaled = image_rescaled.astype(np.uint8)
plt.imshow(image_rescaled)


thin1 = morphology.thin(image_rescaled, max_iter=3)
plt.imshow(~thin1, cmap='gray')


skel = skeletonize(edge_frangi >= .3)
plot_comparison(edge_frangi, skel, 'skel')

dilated = erosion(image_rescaled)
plot_comparison(edge_frangi, dilated, 'dilation')




closed = closing(gray0)
plot_comparison(gray0, closed, 'closing')

filled = ndimage.binary_fill_holes(closed).astype(int)
plt.imshow(filled)



edge_prewitt = filters.prewitt(gray0)
plt.imshow(edge_prewitt, cmap='gray')

plot_comparison(edge_scharr, edge_prewitt, 'closing')


# Enhance RGB
enh0 = EnhanceImage(image2, Color = 3, Contrast = None, Sharp = None)
plt.imshow(enh0)
plt.imshow(image1[:, :, 1]>100)


# Convert to gray
gray0 = enh0 @ [0.2126, 0.7152, 0.0722]
io.imshow(gray0, cmap='gray')



# Enhance gray
img = gray0

# Equalization
img_eq = exposure.equalize_hist(img)
plt.imshow(img_eq, cmap='gray')

# Contrast stretching
p2, p98 = np.percentile(img_eq, (2, 98))
img_rescale = exposure.rescale_intensity(img_eq, in_range=(p2, p98))
plt.imshow(img_rescale, cmap='gray')


# fig, ax = try_all_threshold(img_rescale, figsize=(10, 8), verbose=False)
# plt.show()



edges = filters.sobel(gray0)
edges2 = filters.sobel(img_rescale)
plot_comparison(edges, result >= 0.08, "something")

edges3=edges2 >= 0.35


from scipy.ndimage import gaussian_filter
result = gaussian_filter(edges2, sigma=3)
# closed = closing(edges3)
plot_comparison(edges3, result, 'closing')

image_rescaled = rescale(edges3[...], 0.25, preserve_range=True, multichannel=False)
plt.imshow(image_rescaled)
final_image = image_rescaled.astype(np.uint8)
image1 = final_image




edges3 = filters.sobel(edges2)
plot_comparison(edges3, edges2, "something")


### Fin edges with PIL
from PIL import Image, ImageFilter
from matplotlib import cm
gray_pil = Image.fromarray(np.uint8(cm.gist_earth(img_rescale)*255))
edges_pil = gray_pil.convert("L")
edges_pil = edges_pil.filter(ImageFilter.FIND_EDGES)


# Combined PIL's and Sobel's edges
edges_pil = np.array(edges_pil)
testing = edges_pil/edges_pil.max()
plot_comparison(edges_pil, testing>=.5, "something")
# plt.imshow(edges_pil, cmap = 'gray')
combined_gray = edges_pil * edges2
plt.imshow(combined_gray, cmap='gray')


# Median filter
from scipy import ndimage, misc
import matplotlib.pyplot as plt

result = ndimage.median_filter(edges2, size=(15,15))
plot_comparison(edges2, result, "something")

