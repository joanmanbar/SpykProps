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



# # Gather the image files (change path)
# Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\Images\IN_DIR\*.tif')
ImagePath = Images.files[0]

Spks_data = pd.DataFrame()

# Define the function
def SpikeProps(Im):
    
    Im = Images.files[0]
    image_time = time.time()
    spks_df = SpikesDF(Im)
    print("The image took", time.time() - image_time, "seconds to run.")
    
    
    
    lab = color.rgb2lab(rgb)
    
    
    
    # Label folder of spikes
    ImagesDir = "J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\Images\IN_DIR"
    PrintSpkLabel(ImagesDir, ReduceImageSize = True, Save = True)
    
    
    Spks_data = Spks_data.append(spks_df, ignore_index=True)
    
       
        
        
        # Edge detector
        edge_frangi = filters.frangi(gray0, black_ridges=False)
        
        # Resize
        image_rescaled = rescale(edge_frangi[...], 0.25, preserve_range=True, multichannel=False)
        # image_rescaled = image_rescaled.astype(np.uint8)
        # plt.imshow(image_rescaled, cmap='gray')
        
        
        # Skeletonize
        skel0 = morphology.skeletonize(image_rescaled > 0.2)
        # io.imshow(skel0)
        
        # Closing
        closed = morphology.closing(skel0)
        # plt.imshow(closed, cmap='gray')
        
        # Skeletonize
        skel1 = morphology.medial_axis(closed)
        
        
        
        
        # ---- Measure branches ----
    
        # line below is option. Uncomment ONLY if you want to look at the skeleton (degrees)
#        pixel_graph, coordinates, degrees = skeleton_to_csgraph(skel2)
        
        # This one is AMAZING!!
#        io.imshow(degrees)
        

        
        # Measuring the length of skeleton branches
        BranchesPerSpike_all = summarize(Skeleton(skel1))
#        BranchesPerSpike_all.head()
        
#        BranchesPerSpike = BranchesPerSpike_all[['node-id-src', 'node-id-dst', 'branch-distance',      'branch-type', 'coord-src-0', 'coord-src-1', 'coord-dst-0', 'coord-dst-1', 'euclidean-distance']]
        
        # Subset the variables to save computing resources
        BranchesPerSpike = BranchesPerSpike_all[['branch-distance','branch-type','euclidean-distance']]
        
        # Add image name to data frame
        BranchesPerSpike['Image_name'] = i.split('\\')[-1]
        
        # Add spike number to data frame
        BranchesPerSpike['Spike'] = spk
        
        # Append
        Branches_per_image = Branches_per_image.append(BranchesPerSpike)

        
        # Even more amazing!
#        draw.overlay_euclidean_skeleton_2d(myspk, BranchesPerSpike_all, skeleton_color_source='branch-type');
        
        # Histograms
#        BranchesPerSpike.hist(column='branch-distance', by='branch-type', bins=100)
        
        # Export csv (to test only)
#        BranchesPerSpike.to_csv(r'Branches_in_image_v01.csv', header=True)
            
        
        # Create lists
        Images_Names = []
        Spks = []
        Areas = []
        Lengths = []
        Widths = []
        Orientations = []
        Circularitys = []
        Eccentricitys = []
        Rs = []
        Gs = []
        Bs = []
            
        # Loop through the spikes in image     
        for ind,props in enumerate(regions):
            Spk = props.label
            Area = props.area
            Length = props.major_axis_length
            # Length = spike_length
            Width = props.minor_axis_length
            Orientation = props.orientation
            Circularity = (4 * np.pi * props.area) / (props.perimeter ** 2)
            Eccentricity = props.eccentricity
            R =  red_means[ind]
            G =  green_means[ind]
            B =  blue_means[ind]
            
            Image_Name = i
            Image_Name = Image_Name.split('\\')[-1]        
            
            Images_Names.append(Image_Name)
            Spks.append(Spk)
            Areas.append(Area)
            Lengths.append(Length)
            Widths.append(Width)
            Orientations.append(Orientation)
            Circularitys.append(Circularity)
            Eccentricitys.append(Eccentricity)
            Rs.append(R)
            Gs.append(G)
            Bs.append(B)
        
        
        
        
        
        
               
        
        # Watershed
        
        
        # edges = feature.canny(myspk, sigma=1)
        # # io.imshow(edges, cmap = 'gray')
        
        
        # se_de = se0 = morphology.disk(2) 
        # diledges = morphology.dilation(edges, selem=se_de, out=None, shift_x=False, shift_y=False)
        # io.imshow(diledges)
    
        myspk_rot = myspk
        
        distance = ndi.distance_transform_edt(myspk_rot)
        # io.imshow(distance)
        # local_maxi = feature.peak_local_max(distance, indices=False, footprint=morphology.diamond(30), labels=myspk_rot)
        local_maxi = feature.peak_local_max(distance, indices=False, min_distance=50, labels=myspk_rot)
        stem = morphology.remove_small_objects(local_maxi, min_size=5)
        # io.imshow(img_as_float(local_maxi) - img_as_float(stem))
        
        new_local_max = img_as_float(local_maxi) - img_as_float(stem)
        new_local_max = new_local_max.astype(np.bool)
        
        # local_maxi = feature.corner_peaks(distance, indices=False, min_distance=20, labels=myspk_rot)
        # io.imshow(new_local_max)
        
        
        
        markers = ndi.label(new_local_max)[0]
        labeled_spikelets = segmentation.watershed(-distance, markers, mask=myspk_rot)
        plt.imshow(labeled_spikelets)
        
        regions_spikelets = regionprops(labeled_spikelets)
        
        # n_Spikelets = int(labeled_spikelets[:,:].max())
        
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        
        ax[0].imshow(myspk_rot, cmap=plt.cm.gray)
        ax[0].set_title('Overlapping objects')
        ax[1].imshow(-distance, cmap=plt.cm.gray)
        ax[1].set_title('Distances')
        ax[2].imshow(labeled_spikelets, cmap=plt.cm.nipy_spectral)
        ax[2].set_title('Separated objects')
        
        for a in ax:
            a.set_axis_off()
        
        fig.tight_layout()
        plt.show()
        
        # Create lists
        s_Images_Names = []
        s_Spks = []
        s_Areas = []
        s_Lengths = []
        s_Widths = []
        s_Orientations = []
        s_Circularitys = []
        s_Eccentricitys = []
        s_Rs = []
        s_Gs = []
        s_Bs = []
            
        # Loop through the spikes in image     
        for ind,props in enumerate(regions_spikelets):
            Spk = props.label
            Area = props.area
            Length = props.major_axis_length
            # Length = spike_length
            Width = props.minor_axis_length
            Orientation = props.orientation
            Circularity = (4 * np.pi * props.area) / (props.perimeter ** 2)
            Eccentricity = props.eccentricity
            R =  red_means[ind]
            G =  green_means[ind]
            B =  blue_means[ind]
            
            Image_Name = i
            Image_Name = Image_Name.split('\\')[-1]        
            
            s_Images_Names.append(Image_Name)
            s_Spks.append(Spk)
            s_Areas.append(Area)
            s_Lengths.append(Length)
            s_Widths.append(Width)
            s_Orientations.append(Orientation)
            s_Circularitys.append(Circularity)
            s_Eccentricitys.append(Eccentricity)
            s_Rs.append(R)
            s_Gs.append(G)
            s_Bs.append(B)
        
        
 

        
        # Dataframe 1: for single obervation per spike
        Spikes_per_image = pd.DataFrame(list(zip(Images_Names, Spks, Areas, Lengths, Widths, Orientations, Circularitys, Eccentricitys, Rs, Gs, Bs)), columns = ['Image_Name', 'Spike', 'Area', 'Length', 'Width', 'Orientation', 'Circularity', 'Eccentricity', 'Red_mean', 'Green_mean', 'Blue_mean'])
        
        # Append
        Branches_per_image = Branches_per_image.append(BranchesPerSpike)
        
        # Append
        Spikelets_per_image = Spikelets_per_image.append(BranchesPerSpike)
            
        # Return dataset 1 and dataset 2 per image
        return(Spikes_per_image, Branches_per_image)
    
    
    
    # # Create a completely empty dataframe for branches in spikes in image
    # Branches_per_image = pd.DataFrame()
    
    # # Create a completely empty dataframe for spikelets in spike in image
    # Spikelets_per_image = pd.DataFrame()    
    
    
    
    
    
    
    # End of function
#
#






















# Requires a folder with images of spikes!
# Requires an output folder named "Output"
    
# Gather the image files (change path)
Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\*.tif')

# Create a completely empty dataframe for branches in spikes, in images, in folder (dataset 1)
Spikes_data = pd.DataFrame()

# Create a completely empty dataframe for branches in spikes, in images, in folder (dataset 2)
Branches_data = pd.DataFrame()

# Create a completely empty dataframe for branches in spikes, in images, in folder (dataset 2)
Spikelets_data = pd.DataFrame()


# Loop through images in folder
for i in Images.files:
    
    # Set the initial time per image
    image_time = time.time()
    
    # Return the two datasets from the function
    Spikes, Branches = SpikeProps(i)
    
    # How long did it take to run this image?
    print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
     
    # Append to each data set       
    Branches_data = Branches_data.append(Branches)
    Spikes_data = Spikes_data.append(Spikes)
    
    


# Export Branches_data to csv
Branches_data.to_csv (r'Output\Branches_data_3.csv', header=True, index=False)

# Export Branches_data to csv
Spikes_data.to_csv (r'Output\Spikes_data_3.csv', header=True, index=False)


# How long did it take to run the whole code?
print("This entire code took", time.time() - start_time, "seconds to run.")




