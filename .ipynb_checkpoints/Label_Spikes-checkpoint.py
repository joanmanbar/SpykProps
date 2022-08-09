# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:00:13 2021

@author: jbarreto
"""


# Label Spikes

import time
start_time = time.time()

import os
import numpy as np

import matplotlib.pyplot as plt

from skimage import io
import imageio as iio

from skimage.measure import label, regionprops
from skimage import morphology
from skimage.transform import rescale

from datetime import date

# today = str(date.today())
# print("Today's date:", today)


# # Gather the image files (change path)
# Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\*.tif')


# Define the function
def label_spikes(Im, OutDir):
    
      
    # Read image
    # Im = Images.files[1]      
    image0 = iio.imread(Im)
    # io.imshow(image0)
    
    # Image Name
    Image_Name = Im.split('\\')[-1]
    Image_Name = Image_Name.split('.')[-0]
    # print(Image_Name.replace('.tif', ''))
    
    # Crop image based on scanner's margins
    image1 = image0[44:6940, 25:4970, :]
    # io.imshow(image1)
    image_rescaled = rescale(image1[...], 0.25, preserve_range=True, multichannel=True)
    final_image = image_rescaled.astype(np.uint8)
    image1 = final_image
#    io.imshow(image1)

    # Assign each color channel to a different variable
    red = image1[:, :, 0]
    # green = image1[:, :, 1]
    # blue = image1[:, :, 2]
    
    # Threshold based on the red channel (this depends on the image's background)
    bw0 = red > 40
#    io.imshow(bw0)  

    ## Remove noise
    bw1 = morphology.remove_small_objects(bw0, min_size=5000) # Filter out objects whose area < 10000
#    io.imshow(bw1)
    
    # Apply mask to RGB
    # image2 = np.asarray(image1)
    # image2 = np.where(bw1[..., None], image1, 0)        # third condition changes mask's background on a 1-255 scale
#    io.imshow(image2)   


   # ------------ REGIONPROPS ------------
    
    # Import label and regionprops
    
    # Label spikes
    labeled_spks, num_spikes = label(bw1, return_num = True)
    #io.imshow(labeled_spks == 2)
    
    # Visualize labels
    # io.imshow(labeled_spks)     # less computing intensive
    # image_label_overlay = color.label2rgb(labeled_spks, image=bw1, bg_label=0)  # more computing intensive
    # io.imshow(image_label_overlay)
        
    regions = regionprops(labeled_spks)
    
    # Verify that "Labeled" folder exists
    OUTpath = str(OutDir + '\Labeled')
    if not os.path.exists(OUTpath):
        os.makedirs(OUTpath)
    
    plt.ioff()
    fig, ax = plt.subplots()
    ax.imshow(bw1, cmap=plt.cm.gray)
    spike_ind = 0
    
    for props in regions:
        y0, x0 = props.centroid
        spike_ind = spike_ind + 1
        plt.text(x0, y0, str(spike_ind), color="red", fontsize=15)
  
    # plt.show()
    fig_name = str(OUTpath + '\\' + str(Image_Name) + '.png')
    plt.savefig(fig_name, dpi=300)
    # plt.close(fig)


Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\*.tif')
OutDir = str(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images')
# Loop through images in folder
for i in Images.files:
    
    # Set the initial time per image
    image_time = time.time()
    
    # Label spikes
    label_spikes(i, OutDir = OutDir)
    
    # How long did it take to run this image?
    print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
    




# How long did it take to run the whole code?
print("This entire code took", time.time() - start_time, "seconds to run.")










