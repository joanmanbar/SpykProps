#!/usr/bin/env python3

# Spike Props v02




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

import glob

# Requires a folder with images of spikes!
# Requires an output folder named "Output"
    
# Gather the image files (change path)
# Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\*.tif')
# Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\Images\IN_DIR\*.tif')
mypath = r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\Images\ShatNurseTROE2020'
Images = glob.glob(mypath + '/**/*.tif', recursive=True)

# Create a completely empty dataframe for branches in spikes, in images, in folder (dataset 1)
Spikes_data = pd.DataFrame()

# Loop through images in folder
for i in Images:
    
    # Set the initial time per image
    image_time = time.time()
    
    # Return the two datasets from the function
    Spikes = SpikesDF(i)
    
    # How long did it take to run this image?
    print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
     
    # Append to data set       
    Spikes_data = Spikes_data.append(Spikes)
    

# Export Branches_data to csv
# Spikes_data.to_csv (r'Output\Spikes_data_06152021.csv', header=True, index=False)


# How long did it take to run the whole code?
print("This entire code took", time.time() - start_time, "seconds to run.")




