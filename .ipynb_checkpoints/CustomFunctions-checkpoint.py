#!/usr/bin/env python3


#-----------------------------------------------------#
#               CUSTOM FUNCTIONS
#-----------------------------------------------------#
#
# This .py file contains the definitions to some useful custom functions
#
#
#








# Dependencies

import time
# start_time = time.time()

import os

import numpy as np

import matplotlib.pyplot as plt

from skimage import io, feature, filters, color, util, morphology, exposure, segmentation, img_as_float
from skimage.filters import unsharp_mask
from skimage.measure import label, regionprops, perimeter, find_contours
from skimage.morphology import medial_axis, skeletonize, convex_hull_image, binary_dilation, black_tophat, diameter_closing, area_opening, erosion, dilation, opening, closing, white_tophat, reconstruction, convex_hull_object
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.util import invert

from glob import glob

import cv2

import imageio as iio

from PIL import Image, ImageEnhance, ImageCms

# import scipy
# from scipy import ndimage as ndi

# from PIL import Image, ImageEnhance 

# import cv2 as cv

# import math

# import pandas as pd                          

# Random label cmap
import matplotlib
import colorsys

import pandas as pd

from scipy import ndimage

# from fil_finder import FilFinder2D

# import astropy.units as u

# from skan import Skeleton, summarize

import scipy
from scipy import ndimage as ndi

import seaborn as sns



#-----------------------------------------------------#
#               Random Label cmap
#-----------------------------------------------------#
#
# Found here: https://github.com/matplotlib/matplotlib/issues/16976/
# Used for StarDist
#
#

def random_label_cmap(n=2**16):

    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

lbl_cmap = random_label_cmap()






#-----------------------------------------------------#
#               Cropping Function
#-----------------------------------------------------#
# This function takes an rgb or gray image, or the image's fullpath, and crops it to the area where the stems are clustered.
#
#
# # For debugging
# # Images folder (change extension if needed)
# Images = io.ImageCollection(r'..\Images\Stems\*.JPG')
# rgb = Images.files[0]


def CropStems(InputImage, rgb_out = False, sigma = 0.9, low_threshold = 0, high_threshold = .75):
    
    # Read image
    if isinstance(InputImage, str) == True:
        rgb = iio.imread(InputImage)
        gray0 = rgb @ [0.2126, 0.7152, 0.0722]
    elif len(InputImage.shape) == 3:
        rgb = InputImage
        gray0 = rgb @ [0.2126, 0.7152, 0.0722]
    else: 
        gray0 = InputImage
    
    # Normalize
    gray0 = gray0/255
    
    # Detect edges
    edges = feature.canny(gray0, sigma = sigma, low_threshold = low_threshold, high_threshold = high_threshold)
    # plt.imshow(edges, cmap = 'gray')
    
    # Dilate
    dilated = binary_dilation(edges, selem=morphology.diamond(10), out=None)
    # plt.imshow(dilated, cmap = 'gray')
    
    # Get convex hull
    chull = convex_hull_object(dilated, connectivity=2)
    # plt.imshow(chull)
    
    cropped = np.asarray(chull)
    
    if rgb_out == False:
        cropped = np.where(chull, gray0, 0)
        cropped = cropped*255
    else:
        cropped = np.where(chull[..., None], rgb, 0)

    cropped = cropped.astype(np.uint8)
    # Crop image
    [rows, columns] = np.where(chull)
    row1 = min(rows)
    row2 = max(rows)
    col1 = min(columns)
    col2 = max(columns)
    cropped = cropped[row1:row2, col1:col2]
    
    # plt.imshow(cropped)
    
    return cropped


#-----------------------------------------------------#
#               Enhance Image
#-----------------------------------------------------#
# This function takes an rgb or gray image and changes the color and/or sharpeness values based on Image and ImageEnhance from PILLOW
# Doc: https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html
# Examples: https://medium.com/swlh/image-enhance-recipes-in-pillow-67bf39b63bd
#
#


def EnhanceImage(InputImage, Color = None, Contrast = None, Sharp = None):
    
    # Read image
    if isinstance(InputImage, str) == True:
        img = iio.imread(InputImage)
    else: 
        img = InputImage
    
    # RGB enhancement
    img0 = Image.fromarray(img)
    
    # Color seems to be good around 3.5
    img1 = ImageEnhance.Color(img0)
    if Color is not None:
        img1 = img1.enhance(Color)
    else:
        img1 = img0
    
    # Contrast
    img2 = ImageEnhance.Contrast(img1)
    if Contrast is not None:
        img2 = img2.enhance(Contrast)
    else:
        img2 = img1
    
    # Sharpness (Good ~20 or higher)
    img3 = ImageEnhance.Sharpness(img2)    
    if Sharp is not None:
        img3 = img3.enhance(Sharp)
    else:
        img3 = img2
    
    # Final image
    img3 = np.array(img3)
    
    return img3
    


#-----------------------------------------------------#
#               Compare two plots
#-----------------------------------------------------#
# Taken from scikit-image: 
# Link: https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html
#
#


def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')







#-----------------------------------------------------#
#               Remove spike's background
#-----------------------------------------------------#
#
# Function to remove background of spikes from 2019
# Returns an rgb image

def RemoveBackground(img, OtsuScaling=0.25, rgb_out=True):
    
    # Read image
    if isinstance(img, str) == True:
        img0 = iio.imread(img)
    else: 
        img0 = img
    
    img1 = img0[44:6940, 25:4970, :]
    
     # Convert to gray
    gray0 = img1 @ [0.2126, 0.7152, 0.0722]
    
    # Set image threshold
    T = filters.threshold_otsu(gray0)
#     print(T)
    T = T*OtsuScaling
#     print(T)
    
    # Segment gray image
    bw0 = gray0 > T
    
    # Remove small objects
    n_pixels = gray0.shape[0] * gray0.shape[1]
    minimum_size = n_pixels/10000
    bw1 = morphology.remove_small_objects(bw0, min_size=np.floor(minimum_size))
    
    if rgb_out==True:
        # Apply mask to RGB
        img2 = np.where(bw1[..., None], img1, 0)
        return img2












#-----------------------------------------------------#
#               Compare Multiple Plots
#-----------------------------------------------------#
# Compare multiple plots at once. 
#

def ComparePlots(rows, cols, images):
    plots = rows * cols
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(plots):
        ax[i].imshow(images[i], cmap='gray')
        Title = "Image " + str(i)
        ax[i].set_title(Title, fontsize=20)
    fig.tight_layout()
    plt.show()






#-----------------------------------------------------#
#        Channel's Percentile 
#-----------------------------------------------------#
# This function returns a list containing lists with the 25th. 50th, and 75th percentiles, for a given color channel
#
#


def channel_percentiles(channel_props, Negatives = None):
      
    # Create empty lists to populate    
    p25_pos_list = []
    p50_pos_list = []
    p75_pos_list = []
    p25_neg_list = []
    p50_neg_list = []
    p75_neg_list = []
    mean_pos_list = []
    sd_pos_list = []
    min_pos_list = []
    max_pos_list = []
    mean_neg_list = []
    sd_neg_list = []
    min_neg_list = []
    max_neg_list = []
    
    
    # channel_props = red_props
    for spk in range(len(channel_props)):
        my_array = channel_props[spk].intensity_image
        flat_array = my_array.ravel()
        non_zero = flat_array[flat_array != 0]
        
        positive_values = non_zero[non_zero > 0]        
        p25_pos = np.percentile(positive_values, 25)
        p25_pos_list.append(p25_pos)
        p50_pos = np.percentile(positive_values, 50)
        p50_pos_list.append(p50_pos)
        p75_pos = np.percentile(positive_values, 75)
        p75_pos_list.append(p75_pos)
        mean_pos = np.mean(positive_values)
        mean_pos_list.append(mean_pos)
        sd_pos = np.std(positive_values)
        sd_pos_list.append(sd_pos)
        min_pos = min(positive_values)
        min_pos_list.append(min_pos)
        max_pos = max(positive_values)
        max_pos_list.append(max_pos)
        
        if Negatives == True:
            negative_values = non_zero[non_zero < 0]
            p25_neg = np.percentile(negative_values, 25)
            p25_neg_list.append(p25_neg)
            p50_neg = np.percentile(negative_values, 50)
            p50_neg_list.append(p50_neg)
            p75_neg = np.percentile(negative_values, 75)
            p75_neg_list.append(p75_neg)
            mean_neg = np.mean(negative_values)
            mean_neg_list.append(mean_neg)
            sd_neg = np.std(positive_values)
            sd_neg_list.append(sd_neg)
            min_neg = min(negative_values)
            min_neg_list.append(min_neg)
            max_neg = max(negative_values)
            max_neg_list.append(max_neg)
            
            Lists = [p25_pos_list, p50_pos_list, p75_pos_list, mean_pos_list, sd_pos_list, min_pos_list, max_pos_list, p25_neg_list, p50_neg_list, p75_neg_list, mean_neg_list, sd_neg_list, min_neg_list, max_neg_list]
        else:
            Lists = [p25_pos_list, p50_pos_list, p75_pos_list, mean_pos_list, sd_pos_list, min_pos_list, max_pos_list]

        # if any(pixel < 0 for pixel in non_zero) == True:
        #     negative_values = non_zero[non_zero < 0]
        #     p25_neg = np.percentile(negative_values, 25)
        #     p25_neg_list.append(p25_neg)
        #     p50_neg = np.percentile(negative_values, 50)
        #     p50_neg_list.append(p50_neg)
        #     p75_neg = np.percentile(negative_values, 75)
        #     p75_neg_list.append(p75_neg)
    return Lists







#-----------------------------------------------------#
#        Spike's length (approximate) 
#-----------------------------------------------------#
# This function returns a numpy array with the longest lengths of the skeletonized elements in image
#
#

# def SpkLength(bw1):
    
#     # Dilation
#     closed = morphology.closing(bw1, selem=morphology.disk(5), out=None)
#     # io.imshow(dilated)
#     mask = ndimage.binary_fill_holes(closed).astype(int)
#     # plt.imshow(mask)
    
#     # Skeleton
#     skel = morphology.medial_axis(mask)
#     # plt.imshow(skel)
    
#     fil = FilFinder2D(skel, distance=250 * u.pc, mask=skel)
#     fil.preprocess_image(flatten_percent=85)
#     fil.create_mask(border_masking=True, verbose=False,use_existing_mask=True)
#     fil.medskel(verbose=False)
#     fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
    
#     Lengths = fil.skeleton_longpath
#     Labels = label(Lengths)
#     props = regionprops(label(Labels))
#     Lengths = [rp.area for rp in props]
#     Lengths = np.array(Lengths)
    
#     return Lengths

#     #   # Show the longest path
#     # plt.imshow(fil.skeleton, cmap='gray')
#     # plt.contour(fil.skeleton_longpath, colors='r')
#     # plt.axis('off')
#     # plt.show()












#-----------------------------------------------------#
#        Spike's Geometric and Spectral Properties 
#-----------------------------------------------------#
# This function returns a Pandas data frame with the geometric and spectral properties
# of the given path to rgb image
#
#

def SpikesDF(ImagePath, RemoveBG = True, PrintSpkLabels = False):
    
    if RemoveBG == True:
        # Remove background
        img0 = RemoveBackground(ImagePath)
    else: 
        img0 = plt.imread(ImagePath)
    
    # Get Lab values
    Lab = color.rgb2lab(img0)
    
    # Convert to gray
    gray0 = img0 @ [0.2126, 0.7152, 0.0722]
    
    # Threshold
    otsu = filters.threshold_otsu(gray0)
    bw0 = gray0 > otsu
    bw1 = morphology.remove_small_objects(bw0, min_size=1.5e-05 * gray0.shape[0] * gray0.shape[1])
    
    # Approximate spike lengths
    # Lengths = SpkLength(bw1) 
    
    # Regionprops
    labeled_spks, num_spikes = label(bw1, return_num = True)
    props_spikes = regionprops(labeled_spks)
    
    # Create column with image name
    Image_Name = ImagePath.split('\\')[-1]
    Image_Name = [Image_Name] * num_spikes
    
    # Geometric properties
    Labels = [rp.label for rp in props_spikes]
    Areas = [rp.area for rp in props_spikes]
    MajorAxes = [rp.major_axis_length for rp in props_spikes]
    MinorAxes = [rp.minor_axis_length for rp in props_spikes]
    Orientations = [rp.orientation for rp in props_spikes]
    Perimeters = [rp.perimeter for rp in props_spikes]
    Eccentricities = [rp.eccentricity for rp in props_spikes]
   
    # Spectral properties
    red_props = regionprops(labeled_spks, intensity_image=img0[:,:,0])
    green_props = regionprops(labeled_spks, intensity_image=img0[:,:,1])
    blue_props = regionprops(labeled_spks, intensity_image=img0[:,:,2])
    L_props = regionprops(labeled_spks, intensity_image=Lab[:,:,0])
    a_props = regionprops(labeled_spks, intensity_image=Lab[:,:,1])
    b_props = regionprops(labeled_spks, intensity_image=Lab[:,:,2])
    
    red = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in red_props])
    green = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in green_props])
    blue = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in blue_props])
    L = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in L_props])
    a = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in a_props])
    b = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in b_props])
    
    Red_Perc = np.array(channel_percentiles(red_props, Negatives=False)).T
    Green_Perc = np.array(channel_percentiles(green_props, Negatives=False)).T
    Blue_Perc = np.array(channel_percentiles(blue_props, Negatives=False)).T
    L_Perc = np.array(channel_percentiles(L_props)).T
    a_Perc = np.array(channel_percentiles(a_props, Negatives=True)).T
    b_Perc = np.array(channel_percentiles(b_props, Negatives=True)).T

    # Dataframe 1: for single obervation per spike
    Spikes_per_image = pd.DataFrame(
    list(zip(Image_Name, Labels, Areas, MajorAxes, MinorAxes, Orientations, Eccentricities, Perimeters, 
             red[:,0], red[:,1], red[:,2], green[:,0], green[:,1], green[:,2], blue[:,0], blue[:,1], blue[:,2], 
             L[:,0], L[:,1], L[:,2], a[:,0], a[:,1], a[:,2], b[:,0], b[:,1], b[:,2], 
             Red_Perc[:,0], Red_Perc[:,1], Red_Perc[:,2], Red_Perc[:,3], Red_Perc[:,4], Red_Perc[:,5], Red_Perc[:,6], 
             Green_Perc[:,0], Green_Perc[:,1], Green_Perc[:,2], Green_Perc[:,3], Green_Perc[:,4], Green_Perc[:,5], Green_Perc[:,6], 
             Blue_Perc[:,0], Blue_Perc[:,1], Blue_Perc[:,2], Blue_Perc[:,3], Blue_Perc[:,4], Blue_Perc[:,5], Blue_Perc[:,6], 
             L_Perc[:,0], L_Perc[:,1], L_Perc[:,2], L_Perc[:,3], L_Perc[:,4], L_Perc[:,5], L_Perc[:,6], 
             a_Perc[:,0], a_Perc[:,1], a_Perc[:,2], a_Perc[:,3], a_Perc[:,4], a_Perc[:,5], a_Perc[:,6], 
             a_Perc[:,7], a_Perc[:,8], a_Perc[:,9], a_Perc[:,10], a_Perc[:,11], a_Perc[:,12], a_Perc[:,13], 
             b_Perc[:,0], b_Perc[:,1], b_Perc[:,2], b_Perc[:,3], b_Perc[:,4], b_Perc[:,5], b_Perc[:,6], 
             b_Perc[:,7], b_Perc[:,8], b_Perc[:,9], b_Perc[:,10], b_Perc[:,11], b_Perc[:,12], b_Perc[:,13])), 
    columns = ['Image_Name', 'Spike_Label', 'Area', 'MajorAxis', 'MinorAxes', 'Orientation', 'Eccentricity', 'Perimeter', 
               'Red_mean', 'Red_min', 'Red_max', 'Green_mean', 'Green_min', 'Green_max', 'Blue_mean', 'Blue_min', 'Blue_max', 
               'L_mean', 'L_min', 'L_max', 'a_mean', 'a_min', 'a_max', 'b_mean', 'b_min', 'b_max', 
               'Red_p25', 'Red_p50', 'Red_p75', 'Red_Mean', 'Red_sd', 'Red_Min', 'Red_Max', 
               'Green_p25', 'Green_p50', 'Green_p75', 'Green_Mean', 'Green_sd', 'Green_Min', 'Green_Max', 
               'Blue_p25', 'Blue_p50', 'Blue_p75', 'Blue_Mean', 'Blue_sd', 'Blue_Min', 'Blue_Max', 
               'L_p25', 'L_p50', 'L_p75', 'L_Mean', 'L_sd', 'L_Min', 'L_Max', 
               'a_p25_pos', 'a_p50_pos', 'a_p75_pos', 'a_Mean_pos', 'a_sd_pos', 'a_Min_pos', 'a_Max_pos', 
               'a_p25_neg', 'a_p50_neg', 'a_p75_neg', 'a_Mean_neg', 'a_sd_neg', 'a_Min_neg', 'a_Max_neg', 
               'b_p25_pos', 'b_p50_pos', 'b_p75_pos', 'b_Mean_pos', 'b_sd_pos', 'b_Min_pos', 'b_Max_pos', 
               'b_p25_neg', 'b_p50_neg', 'b_p75_neg', 'b_Mean_neg', 'b_sd_neg', 'b_Min_neg', 'b_Max_neg'])
    
    Spikes_per_image['Circularity'] = (4 * np.pi * Spikes_per_image['Area']) / (Spikes_per_image['Perimeter'] ** 2)
    
    return Spikes_per_image






#-----------------------------------------------------#
#        Branches's Geometric and Spectral? Properties 
#-----------------------------------------------------#
# This function returns a Pandas data frame with the geometric and spectral properties
# of the given path to rgb image
#
#

def BranchesDF(ImagePath, PrintSpkLabels = False):
    
    # Remove background
    img0 = RemoveBackground(ImagePath)
    
    # Get Lab values
    # Lab = color.rgb2lab(img0)
    
    # Convert to gray
    gray0 = img0 @ [0.2126, 0.7152, 0.0722]
    
    # Threshold
    otsu = filters.threshold_otsu(gray0)
    bw0 = gray0 > otsu
    bw1 = morphology.remove_small_objects(bw0, min_size=1.5e-05 * gray0.shape[0] * gray0.shape[1])
    
    # plt.imshow(bw1)
    
    # Dilation
    closed = morphology.closing(bw1, selem=morphology.disk(5), out=None)
    # io.imshow(dilated)
    mask = ndimage.binary_fill_holes(closed).astype(int)
    # plt.imshow(mask)
    
    # Skeleton  
    # skel, distance = morphology.medial_axis(mask, return_distance=True)
    skel = morphology.skeletonize(mask)
    # plt.imshow(skel, cmap='gray')
    
    # Measuring the length of skeleton branches
    BranchesPerSpike = summarize(Skeleton(skel))
    
    # Add image name to data frame
    BranchesPerSpike['Image_name'] = ImagePath.split('\\')[-1]
        
    return BranchesPerSpike








#-----------------------------------------------------#
#        Spikelets's Geometric and Spectral Properties 
#-----------------------------------------------------#
# This function returns a Pandas data frame with the geometric and spectral properties
# of the given path to rgb image
#
#

def SpikeletDF(ImagePath, RemoveBG = True, PrintSpkLabels = False):
    
    if RemoveBG == True:
        # Remove background
        img0 = RemoveBackground(ImagePath, 0.25)
    else: 
        img0 = plt.imread(ImagePath)
    
    # Get Lab values
    Lab = color.rgb2lab(img0)
    
    # Convert to gray
    gray0 = img0 @ [0.2126, 0.7152, 0.0722]
    
    # Threshold
    otsu = filters.threshold_otsu(gray0)
    bw0 = gray0 > otsu
    bw1 = morphology.remove_small_objects(bw0, min_size=1.5e-05 * gray0.shape[0] * gray0.shape[1])
    
    # Approximate spike lengths
    # Lengths = SpkLength(bw1) 
    
    # Regionprops
    labeled_spks, num_spikes = label(bw1, return_num = True)
    props_spikes = regionprops(labeled_spks)
    
    # Create column with image name
    Image_Name = ImagePath.split('\\')[-1]
    Image_Name = [Image_Name] * num_spikes
    
    # Geometric properties
    Labels = [rp.label for rp in props_spikes]
    Areas = [rp.area for rp in props_spikes]
    MajorAxes = [rp.major_axis_length for rp in props_spikes]
    MinorAxes = [rp.minor_axis_length for rp in props_spikes]
    Orientations = [rp.orientation for rp in props_spikes]
    Perimeters = [rp.perimeter for rp in props_spikes]
    Eccentricities = [rp.eccentricity for rp in props_spikes]
   
    # Spectral properties
    red_props = regionprops(labeled_spks, intensity_image=img0[:,:,0])
    green_props = regionprops(labeled_spks, intensity_image=img0[:,:,1])
    blue_props = regionprops(labeled_spks, intensity_image=img0[:,:,2])
    L_props = regionprops(labeled_spks, intensity_image=Lab[:,:,0])
    a_props = regionprops(labeled_spks, intensity_image=Lab[:,:,1])
    b_props = regionprops(labeled_spks, intensity_image=Lab[:,:,2])
    
    red = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in red_props])
    green = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in green_props])
    blue = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in blue_props])
    L = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in L_props])
    a = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in a_props])
    b = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in b_props])
    
    Red_Perc = np.array(channel_percentiles(red_props, Negatives=False)).T
    Green_Perc = np.array(channel_percentiles(green_props, Negatives=False)).T
    Blue_Perc = np.array(channel_percentiles(blue_props, Negatives=False)).T
    L_Perc = np.array(channel_percentiles(L_props)).T
    a_Perc = np.array(channel_percentiles(a_props, Negatives=True)).T
    b_Perc = np.array(channel_percentiles(b_props, Negatives=True)).T

    # Dataframe 1: for single obervation per spike
    Spikes_per_image = pd.DataFrame(
    list(zip(Image_Name, Labels, Areas, MajorAxes, MinorAxes, Orientations, Eccentricities, Perimeters, 
             red[:,0], red[:,1], red[:,2], green[:,0], green[:,1], green[:,2], blue[:,0], blue[:,1], blue[:,2], 
             L[:,0], L[:,1], L[:,2], a[:,0], a[:,1], a[:,2], b[:,0], b[:,1], b[:,2], 
             Red_Perc[:,0], Red_Perc[:,1], Red_Perc[:,2], Red_Perc[:,3], Red_Perc[:,4], Red_Perc[:,5], Red_Perc[:,6], 
             Green_Perc[:,0], Green_Perc[:,1], Green_Perc[:,2], Green_Perc[:,3], Green_Perc[:,4], Green_Perc[:,5], Green_Perc[:,6], 
             Blue_Perc[:,0], Blue_Perc[:,1], Blue_Perc[:,2], Blue_Perc[:,3], Blue_Perc[:,4], Blue_Perc[:,5], Blue_Perc[:,6], 
             L_Perc[:,0], L_Perc[:,1], L_Perc[:,2], L_Perc[:,3], L_Perc[:,4], L_Perc[:,5], L_Perc[:,6], 
             a_Perc[:,0], a_Perc[:,1], a_Perc[:,2], a_Perc[:,3], a_Perc[:,4], a_Perc[:,5], a_Perc[:,6], 
             a_Perc[:,7], a_Perc[:,8], a_Perc[:,9], a_Perc[:,10], a_Perc[:,11], a_Perc[:,12], a_Perc[:,13], 
             b_Perc[:,0], b_Perc[:,1], b_Perc[:,2], b_Perc[:,3], b_Perc[:,4], b_Perc[:,5], b_Perc[:,6], 
             b_Perc[:,7], b_Perc[:,8], b_Perc[:,9], b_Perc[:,10], b_Perc[:,11], b_Perc[:,12], b_Perc[:,13])), 
    columns = ['Image_Name', 'Spike_Label', 'Area', 'MajorAxis', 'MinorAxes', 'Orientation', 'Eccentricity', 'Perimeter', 
               'Red_mean', 'Red_min', 'Red_max', 'Green_mean', 'Green_min', 'Green_max', 'Blue_mean', 'Blue_min', 'Blue_max', 
               'L_mean', 'L_min', 'L_max', 'a_mean', 'a_min', 'a_max', 'b_mean', 'b_min', 'b_max', 
               'Red_p25', 'Red_p50', 'Red_p75', 'Red_Mean', 'Red_sd', 'Red_Min', 'Red_Max', 
               'Green_p25', 'Green_p50', 'Green_p75', 'Green_Mean', 'Green_sd', 'Green_Min', 'Green_Max', 
               'Blue_p25', 'Blue_p50', 'Blue_p75', 'Blue_Mean', 'Blue_sd', 'Blue_Min', 'Blue_Max', 
               'L_p25', 'L_p50', 'L_p75', 'L_Mean', 'L_sd', 'L_Min', 'L_Max', 
               'a_p25_pos', 'a_p50_pos', 'a_p75_pos', 'a_Mean_pos', 'a_sd_pos', 'a_Min_pos', 'a_Max_pos', 
               'a_p25_neg', 'a_p50_neg', 'a_p75_neg', 'a_Mean_neg', 'a_sd_neg', 'a_Min_neg', 'a_Max_neg', 
               'b_p25_pos', 'b_p50_pos', 'b_p75_pos', 'b_Mean_pos', 'b_sd_pos', 'b_Min_pos', 'b_Max_pos', 
               'b_p25_neg', 'b_p50_neg', 'b_p75_neg', 'b_Mean_neg', 'b_sd_neg', 'b_Min_neg', 'b_Max_neg'])
    
    Spikes_per_image['Circularity'] = (4 * np.pi * Spikes_per_image['Area']) / (Spikes_per_image['Perimeter'] ** 2)
    
    return Spikes_per_image







#-----------------------------------------------------#
#                       Spikelets 
#-----------------------------------------------------#
# This function returns the number of spikelets and labeled spike if desired.
#
#


def nSpklts(cropped_rgb, labeled_out=False):
    
    # Rescale to 10% of original
    rescaled_spk = rescale(cropped_rgb[...], 0.1, preserve_range=False, multichannel=True, anti_aliasing=True)
    # plt.imshow(rescaled_spk)

    # Erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    erosion = cv2.erode(rescaled_spk,kernel,iterations = 1)
    # plt.imshow(erosion)

    # Opening
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations = 10)
    # plt.imshow(opening)

    # Resize
    rescaled_spk2 = Image.fromarray((rescaled_spk * 255).astype(np.uint8))
    rescaled_spk2 = rescaled_spk2.resize((cropped_rgb.shape[1],cropped_rgb.shape[0]))
    # plt.imshow(rescaled_spk2)
    # rescaled_spk2.size
    opening = np.asarray(rescaled_spk2)

    # Convert rgb to gray
    gray_spklts = opening @ [0.2126, 0.7152, 0.0722]
    # plt.imshow(gray_spklts)

    # Binarize gray
    bw_spklts = gray_spklts > 0
    # plt.imshow(bw_spklts)

    # Get distances
    distance = ndi.distance_transform_edt(bw_spklts)
    # plt.imshow(-distance)

    # Get max peaks
    coords = peak_local_max(distance, min_distance=50, labels=bw_spklts)
    # plt.imshow(coords)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, spikelets = ndi.label(mask)
    # markers, spikelets = label(mask, return_num = True)
    # markers64 = np.int64(markers)

    if labeled_out==True:
    # Watershed
        labels = watershed(-distance, markers, mask=cropped_spk)

        # Detected spikelets
        print('Detected spikelets = ', spikelets)

        # Plot
        plt.imshow(labels, cmap=plt.cm.nipy_spectral)

    else:
        return spikelets






#-----------------------------------------------------#
#        Plot Channel Histograms for Spikes
#-----------------------------------------------------#
# This function returns the black and white version of the given image(s)
# and print the label of each spike as computed by 'skimage.measure.label'
#
#


def PixelHist(bw1, Lab, channel = 0, nbins = 100):
    
    labeled_spks, num_spikes = label(bw1, return_num = True)
    Props = regionprops(labeled_spks, intensity_image=Lab[:,:,channel])
    Areas = [rp.area for rp in Props]
    Spikes = []
    Names = []
    Colors = sns.color_palette("husl", num_spikes)
    Colors2 = [list(i) for i in Colors] # list of lists
    
    for spk in range(num_spikes):
        spk_data = Props[spk].intensity_image
        spk_data = spk_data.ravel()
        spk_data = spk_data[spk_data != 0]
#         spk_data = spk_data/Areas[spk]
#         spk_data = spk_data/len(spk_data)
        Spikes.append(spk_data)
        Names.append("Spike " + str(spk+1) + "\n" + "Area = "  + str(Areas[spk]) + " px" + "\n" +
                     "Mean = "  + str(round(np.mean(spk_data), 1)))
        Colors.append(list(np.random.choice(range(2), size=3)))
    
    plt.hist(Spikes, bins = nbins, color = Colors2, label = Names);
    
    # Plot formatting
    plt.legend();
    plt.xlabel('Intensity Value');
    plt.ylabel('Number of NonZero Pixels');
    plt.title('Histograms of Pixel Values for a Given Channel across Spikes');





    
    
    
    
    
    
    
    
    
    
    
    
#-----------------------------------------------------#
#        Separate spikes into individual images 
#-----------------------------------------------------#
# This function returns as many individual images as labeled in the mask.
#
#



def SeparateSpikes(Image_Path, Outfile = None):
    
    # Remove bakground
    img0 = RemoveBackground(Image_Path)
    
    # Convert to gray
    gray0 = img0 @ [0.2126, 0.7152, 0.0722]

    # Threshold
    otsu = filters.threshold_otsu(gray0)
    bw0 = gray0 > otsu
    bw1 = morphology.remove_small_objects(bw0, min_size=1.5e-05 * gray0.shape[0] * gray0.shape[1])
    
    # Label spikes
    labeled_spks, num_spikes = label(bw1, return_num = True)
    
    # Loop through spikes
    for spk in range(1,num_spikes):
        
        # Select current spike
        myspk = labeled_spks == spk

        # Crop spike
        slice_x, slice_y = ndimage.find_objects(myspk)[0]
        cropped_spk = myspk[slice_x, slice_y]
        
        # Add 100 pixels to each border
        padded = np.pad(cropped_spk, ((100,100), (100,100)))
        
        # Save image 
        im = Image.fromarray(padded)
        
        if Outfile == None:
            
            Split_Path = Image_Path.split("\\")
            OutName = Split_Path[-1].replace(".tif", "")
            Split_Path = Split_Path[:-1]
            OutDir = '\\'.join([str(i) for i in Split_Path])
            OutDir = OutDir + "\\IndividualSpikes\\"
            path = pathlib.Path(OutDir)
            path.mkdir(parents=True, exist_ok=True)
            
            if spk < 10:
                OutName = OutDir + OutName + "_spk0" + str(spk) + '.jpg'
            else: 
                OutName = OutDir + OutName + "_spk" + str(spk) + '.jpg'
        
        im.save(OutName)














#-----------------------------------------------------#
#        Print Spikes' Labels 
#-----------------------------------------------------#
# This function returns the black and white version of the given image(s)
# and print the label of each spike as computed by 'skimage.measure.label'
#
#


def PrintSpkLabel(ImagesDir, ReduceImageSize = True, Save = True):
    
    # Verify output filename/path is given
    # assert OutDir != None, "Provide output name or path"
    OUTpath = str(ImagesDir + '\Labeled')
    ImagesDir = io.ImageCollection(ImagesDir + '\\*.tif')
    
    for i in ImagesDir.files:
        img0 = iio.imread(i)
        bw = RemoveBackground(img0, MaskOnly = True)
        Image_Name = i.split('\\')[-1]
 
        if ReduceImageSize == True:
            image_rescaled = rescale(bw[...], 0.25, preserve_range=True, multichannel=False)
            final_image = image_rescaled.astype(np.uint8)
            bw1 = final_image
        else:
            bw1 = bw
        
        labeled_spks, num_spikes = label(bw1, return_num = True)
        regions = regionprops(labeled_spks)
        
        # Verify that "Labeled" folder exists
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
        
        fig_name = str(OUTpath + '\\' + str(Image_Name) + '.png')
        plt.savefig(fig_name, dpi=300)
    
    
    # if Save == True:
    #     fig_name = str(OUTpath + '\\' + str(Image_Name) + '.png')
    #     plt.savefig(fig_name, dpi=300)
    # else: 
    #     plt.show()
    
    # plt.close(fig)



    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    

#-----------------------------------------------------#
#        Spike's length (approximate) 
#-----------------------------------------------------#
# This function returns the number of pixels in a skeletonized spike.
#
#

def spk_length(spk, method='skelblur', Overlay=True, PlotCH=False):
    
    if method=='skelblur':
        # Severly blur the image
        blur = cv2.blur(np.float32(spk),(100,100))
        # Threshold the blur
        thrb = blur > 0.1
        skeleton = skeletonize(thrb)
#         plt.imshow(skeleton)
        
    if method=='chull':
        # Blur the image with a 50x50 kernel
        blur = cv2.blur(np.float32(spk),(50,50))

        # Get convex hull 
        chull = convex_hull_image(blur>0)

        # Perform skeletonization
        image = chull
        skeleton = skeletonize(image)
    #     plt.imshow(skeleton)
    
    # Spike length
    SpkL = cv2.countNonZero(np.float32(skeleton))
    
    if PlotCH == True:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax = axes.ravel()

        ax[0].set_title('Original picture')
        ax[0].imshow(spk, cmap=plt.cm.gray)
        ax[0].set_axis_off()

        ax[1].set_title('Transformed picture')
        ax[1].imshow(chull, cmap=plt.cm.gray)
        ax[1].set_axis_off()

        plt.tight_layout()
        plt.show()
    
    # Visualize overlay?
    if Overlay == True:
        overlay_images = cv2.addWeighted(np.float32(spk),20,np.float32(skeleton),255,0)
        plt.imshow(overlay_images, cmap='gray')
    
    return SpkL




















    


#-----------------------------------------------------#
#        Spikelet contour 
#-----------------------------------------------------#
# This function returns a plot with the detected spikelets as contours
#

def SpkContours(cropped_rgb, ResizeFactor=30, MinSize = 500, plot=True):
    
    # Copy iamge
    OutImage = cropped_rgb.copy()
    
    # Convert to gray
    cropped_gray = color.rgb2gray(cropped_rgb)
    
    # Reduce image size
    im = Image.fromarray((cropped_gray).astype(np.uint8))
    (width, height) = (im.width // ResizeFactor, im.height // ResizeFactor)
    rescaled_spk = im.resize((width, height))

    # Increase to original size
    (width, height) = (im.width, im.height)
    rescaled_spk = rescaled_spk.resize((width, height))
    rescaled_spk = np.asarray(rescaled_spk)

    # Histogram equalization
    rescaled_spk = exposure.equalize_hist(rescaled_spk)

    # Blur with a Gaussian
    blurred = filters.gaussian(rescaled_spk3, sigma=1, preserve_range=True)

    # Adaptative equalization
    blurred = exposure.equalize_adapthist(blurred)

    # Normalize
    blurred = cv2.normalize(blurred, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    blurred = blurred.astype(np.uint8)

    # Find contours at a constant value of 0.8
    
    # Threshold at 80%
    ret, thresh = cv2.threshold(blurred, 0.8*255, 255, 0)
    thresh = np.uint8(thresh)
#     contours = measure.find_contours(blurred,(0.8*255))
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     len(contours)

    # Remove contours smaller than MinSize
    Filtered = [c for c in contours if cv2.contourArea(c) > MinSize]
    
    # Detected spikelets
    print("Detected spikeletes: ", len(Filtered))
    
    if plot==True:
        img = OutImage.copy()
        # Plot all found contours
        plot_contours = cv2.drawContours(img, Filtered, -1, (0,255,0), 10)
        plt.imshow(plot_contours)







        
        
        
        
        
        
        
        
        
        
        
        
        



#-----------------------------------------------------#
#        Spike's angle (approximate) 
#-----------------------------------------------------#
# This function returns the angle of a line with respect to the top left image.
# The line is fitted on an ellipse, that has been also fitted on the contour (detected spikelet)
#
#



def SpkltAng(cropped_rgb, ResizeFactor=30, MinSize = 500, plot=True):
    
    # Copy iamge
    OutImage = cropped_rgb.copy()
    
    # Convert to gray
    cropped_gray = color.rgb2gray(cropped_rgb)
    
    # Reduce image size
    im = Image.fromarray((cropped_gray).astype(np.uint8))
    (width, height) = (im.width // ResizeFactor, im.height // ResizeFactor)
    rescaled_spk = im.resize((width, height))

    # Increase to original size
    (width, height) = (im.width, im.height)
    rescaled_spk = rescaled_spk.resize((width, height))
    rescaled_spk = np.asarray(rescaled_spk)

    # Histogram equalization
    rescaled_spk = exposure.equalize_hist(rescaled_spk)

    # Blur with a Gaussian
    blurred = filters.gaussian(rescaled_spk3, sigma=1, preserve_range=True)

    # Adaptative equalization
    blurred = exposure.equalize_adapthist(blurred)

    # Normalize
    blurred = cv2.normalize(blurred, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    blurred = blurred.astype(np.uint8)

    # Find contours at a constant value of 0.8
    # Threshold at 80%
    ret, thresh = cv2.threshold(blurred, 0.8*255, 255, 0)
    thresh = np.uint8(thresh)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

       
    thresh2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= MinSize:
            thresh2[output == i + 1] = 255
    
#     plt.imshow(thresh2)
    thresh2 = np.uint8(thresh2)
    
    #     contours = measure.find_contours(blurred,(0.8*255))
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     len(contours)

    # Remove contours smallers than MinSize
#     Filtered = [c for c in contours if cv2.contourArea(c) > MinSize]
    
    # Detected spikelets
    print("Detected spikeletes: ", len(contours))
    
    # Create list for slopes
    Slopes = []
    
    if plot==True:
        
#         OutImage = cropped_rgb.copy()
        # Plot all found contours
        OutImage = cv2.drawContours(OutImage, contours, -1, (255,0,0), 15);
        
        for c in range(len(contours)):
            
            ellipse = cv2.fitEllipse(contours[c])
            OutImage = cv2.ellipse(OutImage,ellipse,(255,128,0),10);

            # Fit a line 
            rows,cols = OutImage.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(contours[c], cv2.DIST_L2,0,0.01,0.01);
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            OutImage = cv2.line(OutImage,(cols-1,righty),(0,lefty),(0,255,0),3);
            
            # Slope from tope left, which is is the origin [0,0]
            rise = (0,lefty)[1] - (cols-1,righty)[1]
            run = cols
            Slope = rise/run
            Slopes.append(Slope)
        
        # Plot
        plt.imshow(OutImage)
    
    return Slopes







#-----------------------------------------------------#
#        Object Properties for Spyk Props
#-----------------------------------------------------#
# This function returns the properties of a desired object in a pandas dataframe
# For example ,the contours of the spike, or labels after watershed transform
#
#

def ObjProps(mask, cropped_rgb, cropped_lab, img_name):

    # Regionprops
    labeled_contours, num_contours = label(mask, return_num = True)
    props_contours = regionprops(labeled_contours)

    # # Create column with image name
    Image_Name = img_name.split('\\')[-1]
    Image_Name = [Image_Name] * num_contours

    # Geometric properties
    Labels = [rp.label for rp in props_contours]
    Areas = [rp.area for rp in props_contours]
    MajorAxes = [rp.major_axis_length for rp in props_contours]
    MinorAxes = [rp.minor_axis_length for rp in props_contours]
    Orientations = [rp.orientation for rp in props_contours]
    Perimeters = [rp.perimeter for rp in props_contours]
    Eccentricities = [rp.eccentricity for rp in props_contours]

    # Spectral properties
    red_props = regionprops(labeled_contours, intensity_image=cropped_rgb[:,:,0])
    green_props = regionprops(labeled_contours, intensity_image=cropped_rgb[:,:,1])
    blue_props = regionprops(labeled_contours, intensity_image=cropped_rgb[:,:,2])
    L_props = regionprops(labeled_contours, intensity_image=cropped_lab[:,:,0])
    a_props = regionprops(labeled_contours, intensity_image=cropped_lab[:,:,1])
    b_props = regionprops(labeled_contours, intensity_image=cropped_lab[:,:,2])

    red = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in red_props])
    green = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in green_props])
    blue = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in blue_props])
    L = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in L_props])
    a = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in a_props])
    b = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in b_props])

    Red_Perc = np.array(channel_percentiles(red_props, Negatives=False)).T
    Green_Perc = np.array(channel_percentiles(green_props, Negatives=False)).T
    Blue_Perc = np.array(channel_percentiles(blue_props, Negatives=False)).T
    L_Perc = np.array(channel_percentiles(L_props)).T
    a_Perc = np.array(channel_percentiles(a_props, Negatives=True)).T
    b_Perc = np.array(channel_percentiles(b_props, Negatives=True)).T

    # Dataframe 1: for single obervation per spike
    contours_per_image = pd.DataFrame(
    list(zip(Image_Name, Labels, Areas, MajorAxes, MinorAxes, Orientations, Eccentricities, Perimeters, 
    red[:,0], red[:,1], red[:,2], green[:,0], green[:,1], green[:,2], blue[:,0], blue[:,1], blue[:,2], 
    L[:,0], L[:,1], L[:,2], a[:,0], a[:,1], a[:,2], b[:,0], b[:,1], b[:,2], 
    Red_Perc[:,0], Red_Perc[:,1], Red_Perc[:,2], Red_Perc[:,3], Red_Perc[:,4], Red_Perc[:,5], Red_Perc[:,6], 
    Green_Perc[:,0], Green_Perc[:,1], Green_Perc[:,2], Green_Perc[:,3], Green_Perc[:,4], Green_Perc[:,5], Green_Perc[:,6], 
    Blue_Perc[:,0], Blue_Perc[:,1], Blue_Perc[:,2], Blue_Perc[:,3], Blue_Perc[:,4], Blue_Perc[:,5], Blue_Perc[:,6], 
    L_Perc[:,0], L_Perc[:,1], L_Perc[:,2], L_Perc[:,3], L_Perc[:,4], L_Perc[:,5], L_Perc[:,6], 
    a_Perc[:,0], a_Perc[:,1], a_Perc[:,2], a_Perc[:,3], a_Perc[:,4], a_Perc[:,5], a_Perc[:,6], 
    a_Perc[:,7], a_Perc[:,8], a_Perc[:,9], a_Perc[:,10], a_Perc[:,11], a_Perc[:,12], a_Perc[:,13], 
    b_Perc[:,0], b_Perc[:,1], b_Perc[:,2], b_Perc[:,3], b_Perc[:,4], b_Perc[:,5], b_Perc[:,6], 
    b_Perc[:,7], b_Perc[:,8], b_Perc[:,9], b_Perc[:,10], b_Perc[:,11], b_Perc[:,12], b_Perc[:,13])), 
    columns = ['Image_Name', 'Spklt_Label', 'Area', 'MajorAxis', 'MinorAxes', 'Orientation', 'Eccentricity', 'Perimeter', 
      'Red_mean', 'Red_min', 'Red_max', 'Green_mean', 'Green_min', 'Green_max', 'Blue_mean', 'Blue_min', 'Blue_max', 
      'L_mean', 'L_min', 'L_max', 'a_mean', 'a_min', 'a_max', 'b_mean', 'b_min', 'b_max', 
      'Red_p25', 'Red_p50', 'Red_p75', 'Red_Mean', 'Red_sd', 'Red_Min', 'Red_Max', 
      'Green_p25', 'Green_p50', 'Green_p75', 'Green_Mean', 'Green_sd', 'Green_Min', 'Green_Max', 
      'Blue_p25', 'Blue_p50', 'Blue_p75', 'Blue_Mean', 'Blue_sd', 'Blue_Min', 'Blue_Max', 
      'L_p25', 'L_p50', 'L_p75', 'L_Mean', 'L_sd', 'L_Min', 'L_Max', 
      'a_p25_pos', 'a_p50_pos', 'a_p75_pos', 'a_Mean_pos', 'a_sd_pos', 'a_Min_pos', 'a_Max_pos', 
      'a_p25_neg', 'a_p50_neg', 'a_p75_neg', 'a_Mean_neg', 'a_sd_neg', 'a_Min_neg', 'a_Max_neg', 
      'b_p25_pos', 'b_p50_pos', 'b_p75_pos', 'b_Mean_pos', 'b_sd_pos', 'b_Min_pos', 'b_Max_pos', 
      'b_p25_neg', 'b_p50_neg', 'b_p75_neg', 'b_Mean_neg', 'b_sd_neg', 'b_Min_neg', 'b_Max_neg'])

    contours_per_image['Circularity'] = (4 * np.pi * contours_per_image['Area']) / (contours_per_image['Perimeter'] ** 2)

    return contours_per_image


































