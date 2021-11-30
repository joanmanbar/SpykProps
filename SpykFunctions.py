#!/usr/bin/env python3


#-----------------------------------------------------#
#                   SpykFunctions
#-----------------------------------------------------#
#
# This .py file contains the definitions to the functions to run SpykProps
#
#
#









# Dependencies

import glob
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import math
import pathlib

from skimage import measure, segmentation, color, filters, morphology, color, feature, io, feature, util, morphology, exposure, img_as_float
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops, perimeter, find_contours
from skimage.future import graph
from skimage.segmentation import watershed, active_contour
from skimage.feature import peak_local_max
from skimage.filters import meijering, sato, frangi, hessian, gaussian
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean, rotate, hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.morphology import medial_axis, skeletonize, convex_hull_image, binary_dilation, black_tophat, diameter_closing, area_opening, erosion, dilation, opening, closing, white_tophat, reconstruction, convex_hull_object
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage.draw import line

from scipy import ndimage
import scipy.ndimage as ndi
import imageio as iio
from PIL import Image, ImageEnhance, ImageCms
from skan import Skeleton, summarize, skeleton_to_csgraph, draw
import seaborn as sns
import imutils
import imageio
import random









def ListImages(path, imgformat=".tif", recursive=False):
    Images = glob.glob(path + '/*' + imgformat, recursive=True)    
    return Images

# Example:
# path = r'./Images/TEST'
# Images = ListImages(path, imgformat=".tif", recursive=False)










def RemoveBackground(img, OtsuScaling=0.25, rgb_out=True, gray_out=True, lab_out=True, hsv_out=True, bw_out=True):
    
    # Read image
    if isinstance(img, str) == True:
        img0 = plt.imread(img)
    else: 
        img0 = img
    
    # Crop images. They were all taken with the same scanner
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
    
    ImagesOut = []
#     len(ImagesOut)
    
    if rgb_out==True:
        # Apply mask to RGB
        rgb = np.where(bw1[..., None], img1, 0)
        ImagesOut.append(rgb)
    if gray_out==True and rgb_out==True:
        gray = color.rgb2gray(rgb)
        ImagesOut.append(gray)
    if lab_out==True and rgb_out==True:
        lab = color.rgb2lab(rgb)
        ImagesOut.append(lab)
    if hsv_out==True and rgb_out==True:
        hsv = color.rgb2hsv(rgb)
        ImagesOut.append(hsv)
    if bw_out==True:
#         # Threshold
#         otsu = filters.threshold_otsu(gray)
#         bw0 = gray > 0
#         bw = morphology.remove_small_objects(bw0, min_size=1.5e-05 * gray.shape[0] * gray.shape[1])
        ImagesOut.append(bw1)
    
    return ImagesOut

# Usage:
# %%time
# I = RemoveBackground(Images[3], OtsuScaling=0.25, rgb_out=True, gray_out=True, lab_out=True, hsv_out=True, bw_out=True)
# rgb0 = I[0]
# gray0 = I[1]
# lab0 = I[2]
# hsv0 = I[3]
# bw0 = I[4]









# Enumerate spikes
def EnumerateSpkCV(bw, rgb, TextSize=5, TROE2020=False, Plot=True, PlotOut=False):
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(bw), connectivity=8)
    img = rgb.copy()
    
    if TROE2020==True:
        counter=-1
    else:
        counter=0
        
    for c in centroids:
#         print(c)
        cx = round(c[0])
        cy = round(c[1])
        img = cv2.circle(img, (cx, cy), 10, (255, 0, 0), -1)
        img = cv2.putText(img, str(counter), (cx - 25, cy - 25),cv2.FONT_HERSHEY_SIMPLEX, TextSize, (255, 0, 0), 15)
        counter = counter+1
    
    if Plot==True:
        plt.imshow(img)
    
    if PlotOut==True:
        return img

# # Example:
# EnumerateSpkCV(bw0, rgb0, TextSize=5, TROE2020=False, Plot=True, PlotOut=False)
# EnumPlot = EnumerateSpkCV(bw0, rgb0, TextSize=5, TROE2020=False, Plot=False, PlotOut=True)
# EnumerateSpkCV(spklts, cropped_rgb, TextSize=5, TROE2020=False)









def spk_length(cropped_spk, method='skelblur', Overlay=True, PlotCH=False):
    
    if method=='skelblur':
        # Severly blur the image
        blur = cv2.blur(np.float32(cropped_spk),(100,100))
        # Threshold the blur
        thrb = blur > 0.1
        skeleton = skeletonize(thrb)
#         plt.imshow(skeleton)
        
    if method=='chull':
        # Blur the image with a 50x50 kernel
        blur = cv2.blur(np.float32(cropped_spk),(50,50))

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
        ax[0].imshow(cropped_spk, cmap=plt.cm.gray)
        ax[0].set_axis_off()

        ax[1].set_title('Transformed picture')
        ax[1].imshow(chull, cmap=plt.cm.gray)
        ax[1].set_axis_off()

        plt.tight_layout()
        plt.show()
    
    # Visualize overlay?
    if Overlay == True:
        overlay_images = cv2.addWeighted(np.float32(cropped_spk),20,np.float32(skeleton),255,0)
        plt.imshow(overlay_images, cmap='gray')
    
    return SpkL

# Example:
# SL = spk_length(cropped_spk, method='skelblur', Overlay=True, PlotCH=False)









def PixelHist(bw, ColorSpace, channel = 0, spikes="All", nbins = 100):
    
    labeled_spks, num_spikes = label(bw, return_num = True)
#     plt.imshow(labeled_spks==0)

    if spikes=="All":
        labeled_spks = labeled_spks
    else:
        for L in range(1,num_spikes+1):
#             print(L)
            if not L in spikes:
#                 print("Deleted label ", L)
                labeled_spks=np.where(labeled_spks==L, 0, labeled_spks)
#     plt.imshow(labeled_spks)
    
    Props = regionprops(labeled_spks, intensity_image=ColorSpace[:,:,channel])
    Areas = [rp.area for rp in Props]
    Labels = [rp.label for rp in Props] #Delete 1 because label in image is +1 greater than ACTUAL label
    Spikes_Data = []
    Names = []
    Colors = sns.color_palette("husl", len(spikes))
    Colors2 = [list(i) for i in Colors] # list of lists
    
    
    for indexed in range(len(Labels)):        
        spk_data = Props[indexed].intensity_image 
        spk_data = spk_data.ravel()
        NonZero = spk_data[spk_data != 0]

        Spikes_Data.append(NonZero)
        Names.append("Spike " + str(int(spikes[indexed])) + "\n" + "Area = "  + str(round(np.mean(NonZero))) + " px" + "\n" +
                     "Mean = "  + str(round(np.mean(NonZero), 1)))
        Colors.append(list(np.random.choice(range(2), size=3)))
    
    plt.hist(Spikes_Data, bins = nbins, color = Colors2, label = Names);
    
    # Plot formatting
    plt.legend();
    plt.xlabel('Intensity Value');
    plt.ylabel('Number of NonZero Pixels');
    plt.title('Distribution of None-Zero Pixel Values for Selected Given Channel and Spikes');

# Example:
# PixelHist(bw=bw0, ColorSpace=lab0, channel = 0, spikes=[1,2,26], nbins = 100)









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
    
#     channel_props[1].intensity_image
    

    # channel_props = a_props
    for spk in range(len(channel_props)):
        # spk=0
        my_array = channel_props[spk].intensity_image
#         plt.imshow(my_array)
        flat_array = my_array.ravel()
        non_zero = flat_array[flat_array != 0]

        positive_values = non_zero[non_zero > 0]
        if positive_values.size == 0:
            positive_values = [0]        
        p25_pos = np.nanpercentile(positive_values, 25)
        p25_pos_list.append(p25_pos)
        p50_pos = np.nanpercentile(positive_values, 50)
        p50_pos_list.append(p50_pos)
        p75_pos = np.nanpercentile(positive_values, 75)
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
            # Make sure list is not empty, otherwise add a zero
            if negative_values.size == 0:
                negative_values = [0]
            p25_neg = np.nanpercentile(negative_values, 25)
            p25_neg_list.append(p25_neg)
            p50_neg = np.nanpercentile(negative_values, 50)
            p50_neg_list.append(p50_neg)
            p75_neg = np.nanpercentile(negative_values, 75)
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


########################################################################################
# Examples:

# For one spike
# labeled_contours, num_contours = label(cropped_spk, return_num = True)
# red_props = regionprops(labeled_contours, intensity_image=cropped_rgb)
# a_Perc = np.array(channel_percentiles(channel_props=a_props, Negatives=True)).T
# len(a_Perc)


# For one entire image
# labeled_contours, num_contours = label(bw0, return_num = True)
# red_props = regionprops(labeled_contours, intensity_image=rgb0)
# Red_Perc = np.array(channel_percentiles(channel_props=red_props, Negatives=False)).T
# len(Red_Perc)









def SpikesDF(I, ImagePath, RemoveBG=False, PrintSpkLabels=False, rm_envelope=False):
    
    # Check if images or path were given
    if RemoveBG == True:
        # Remove background (path was given)
        I = RemoveBackground(ImagePath, OtsuScaling=0.25, rgb_out=True, gray_out=True, lab_out=True, hsv_out=True, bw_out=True)
        rgb0 = I[0]
        gray0 = I[1]
        lab0 = I[2]
        hsv0 = I[3]
        bw0 = I[4]
        
        Image_Name = ImagePath.split('\\')[-1]

    else: 
        # Images were given in a list as returned by RemoveBackground()
        rgb0 = I[0]
        gray0 = I[1]
        lab0 = I[2]
        hsv0 = I[3]
        bw0 = I[4]
    
    
    labeled_spks, num_spikes = label(bw0, return_num = True)
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
    red_props = regionprops(labeled_spks, intensity_image=rgb0[:,:,0])
    green_props = regionprops(labeled_spks, intensity_image=rgb0[:,:,1])
    blue_props = regionprops(labeled_spks, intensity_image=rgb0[:,:,2])
    L_props = regionprops(labeled_spks, intensity_image=lab0[:,:,0])
    a_props = regionprops(labeled_spks, intensity_image=lab0[:,:,1])
    b_props = regionprops(labeled_spks, intensity_image=lab0[:,:,2])
    H_props = regionprops(labeled_spks, intensity_image=hsv0[:,:,0])
    S_props = regionprops(labeled_spks, intensity_image=hsv0[:,:,1])
    V_props = regionprops(labeled_spks, intensity_image=hsv0[:,:,2])
    
    red = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in red_props])
    green = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in green_props])
    blue = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in blue_props])
    L = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in L_props])
    a = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in a_props])
    b = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in b_props])
    H = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in H_props])
    S = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in S_props])
    V = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in V_props])
    
    Red_Perc = np.array(channel_percentiles(red_props, Negatives=False)).T
    Green_Perc = np.array(channel_percentiles(green_props, Negatives=False)).T
    Blue_Perc = np.array(channel_percentiles(blue_props, Negatives=False)).T
    L_Perc = np.array(channel_percentiles(L_props)).T
    a_Perc = np.array(channel_percentiles(a_props, Negatives=True)).T
    b_Perc = np.array(channel_percentiles(b_props, Negatives=True)).T
    H_Perc = np.array(channel_percentiles(H_props)).T
    S_Perc = np.array(channel_percentiles(S_props)).T
    V_Perc = np.array(channel_percentiles(V_props)).T

    # Dataframe 1: for single obervation per spike
    Spikes_per_image = pd.DataFrame(
    list(zip(Image_Name, Labels, Areas, MajorAxes, MinorAxes, Orientations, Eccentricities, Perimeters, 
             red[:,0], red[:,1], red[:,2], green[:,0], green[:,1], green[:,2], blue[:,0], blue[:,1], blue[:,2], 
             L[:,0], L[:,1], L[:,2], a[:,0], a[:,1], a[:,2], b[:,0], b[:,1], b[:,2], 
             H[:,0], H[:,1], H[:,2], S[:,0], S[:,1], S[:,2], V[:,0], V[:,1], V[:,2], 
             Red_Perc[:,0], Red_Perc[:,1], Red_Perc[:,2], Red_Perc[:,3], Red_Perc[:,4], Red_Perc[:,5], Red_Perc[:,6], 
             Green_Perc[:,0], Green_Perc[:,1], Green_Perc[:,2], Green_Perc[:,3], Green_Perc[:,4], Green_Perc[:,5], Green_Perc[:,6], 
             Blue_Perc[:,0], Blue_Perc[:,1], Blue_Perc[:,2], Blue_Perc[:,3], Blue_Perc[:,4], Blue_Perc[:,5], Blue_Perc[:,6], 
             L_Perc[:,0], L_Perc[:,1], L_Perc[:,2], L_Perc[:,3], L_Perc[:,4], L_Perc[:,5], L_Perc[:,6], 
             a_Perc[:,0], a_Perc[:,1], a_Perc[:,2], a_Perc[:,3], a_Perc[:,4], a_Perc[:,5], a_Perc[:,6], 
             a_Perc[:,7], a_Perc[:,8], a_Perc[:,9], a_Perc[:,10], a_Perc[:,11], a_Perc[:,12], a_Perc[:,13], 
             b_Perc[:,0], b_Perc[:,1], b_Perc[:,2], b_Perc[:,3], b_Perc[:,4], b_Perc[:,5], b_Perc[:,6], 
             b_Perc[:,7], b_Perc[:,8], b_Perc[:,9], b_Perc[:,10], b_Perc[:,11], b_Perc[:,12], b_Perc[:,13],
             H_Perc[:,0], H_Perc[:,1], H_Perc[:,2], H_Perc[:,3], H_Perc[:,4], H_Perc[:,5], H_Perc[:,6],
             S_Perc[:,0], S_Perc[:,1], S_Perc[:,2], S_Perc[:,3], S_Perc[:,4], S_Perc[:,5], S_Perc[:,6],
             V_Perc[:,0], V_Perc[:,1], V_Perc[:,2], V_Perc[:,3], V_Perc[:,4], V_Perc[:,5], V_Perc[:,6])), 
    columns = ['Image_Name', 'Spike_Label', 'Area', 'MajorAxis', 'MinorAxes', 'Orientation', 'Eccentricity', 'Perimeter', 
               'Red_mean', 'Red_min', 'Red_max', 'Green_mean', 'Green_min', 'Green_max', 'Blue_mean', 'Blue_min', 'Blue_max', 
               'L_mean', 'L_min', 'L_max', 'a_mean', 'a_min', 'a_max', 'b_mean', 'b_min', 'b_max',
               'H_mean', 'H_min', 'H_max', 'S_mean', 'S_min', 'S_max', 'V_mean', 'V_min', 'V_max', 
               'Red_p25', 'Red_p50', 'Red_p75', 'Red_Mean', 'Red_sd', 'Red_Min', 'Red_Max', 
               'Green_p25', 'Green_p50', 'Green_p75', 'Green_Mean', 'Green_sd', 'Green_Min', 'Green_Max', 
               'Blue_p25', 'Blue_p50', 'Blue_p75', 'Blue_Mean', 'Blue_sd', 'Blue_Min', 'Blue_Max', 
               'L_p25', 'L_p50', 'L_p75', 'L_Mean', 'L_sd', 'L_Min', 'L_Max', 
               'a_p25_pos', 'a_p50_pos', 'a_p75_pos', 'a_Mean_pos', 'a_sd_pos', 'a_Min_pos', 'a_Max_pos', 
               'a_p25_neg', 'a_p50_neg', 'a_p75_neg', 'a_Mean_neg', 'a_sd_neg', 'a_Min_neg', 'a_Max_neg', 
               'b_p25_pos', 'b_p50_pos', 'b_p75_pos', 'b_Mean_pos', 'b_sd_pos', 'b_Min_pos', 'b_Max_pos', 
               'b_p25_neg', 'b_p50_neg', 'b_p75_neg', 'b_Mean_neg', 'b_sd_neg', 'b_Min_neg', 'b_Max_neg',
               'H_p25', 'H_p50', 'H_p75', 'H_Mean', 'H_sd', 'H_Min', 'H_Max',
               'S_p25', 'S_p50', 'S_p75', 'S_Mean', 'S_sd', 'S_Min', 'S_Max',
               'V_p25', 'V_p50', 'V_p75', 'V_Mean', 'V_sd', 'V_Min', 'V_Max'])
    
    Spikes_per_image['Circularity'] = (4 * np.pi * Spikes_per_image['Area']) / (Spikes_per_image['Perimeter'] ** 2)
    
    # Remove envelope's data  
    if rm_envelope==True:
        return Spikes_per_image.iloc[1: , :]
    else:
        return Spikes_per_image


# Example:
# df = SpikesDF(I, ImagePath=img_name, RemoveBG=False, PrintSpkLabels=False)









def SpkltThresh(cropped, ResizeFactor=30, thr2=0.8, MinSize=1000):   
    
    # Check that it's a gray image
    if len(cropped.shape) > 2:
        # Convert to gray
        cropped_gray = color.rgb2gray(cropped)
    else:
        cropped_gray = cropped
        
   
    # Reduce image size
    im = Image.fromarray((cropped_gray*255).astype(np.uint8))
    (width, height) = (im.width // ResizeFactor, im.height // ResizeFactor)
    rescaled_spk = im.resize((width, height))

    # Increase to original size
    (width, height) = (im.width, im.height)
    rescaled_spk = rescaled_spk.resize((width, height))
    rescaled_spk = np.asarray(rescaled_spk)

    # Histogram equalization
    rescaled_spk = exposure.equalize_hist(rescaled_spk)

    # Blur with a Gaussian
    blurred = filters.gaussian(rescaled_spk, sigma=1, preserve_range=True)

    # Adaptative equalization
    blurred = exposure.equalize_adapthist(blurred)

    # Normalize
    blurred = cv2.normalize(blurred, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    blurred = blurred.astype(np.uint8)
    
    if thr2 < 1 == True:
        thr2 = thr2*255
    else:
        thr2 = thr2
    
    # Threshold at given %
    ret, thresh = cv2.threshold(blurred, thr2, 255, 0)
    thresh = np.uint8(thresh)
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1    
       
    thresh2 = np.zeros((output.shape))
    
    # Keep only objects with minimum size
    for i in range(0, nb_components):
        if sizes[i] >= MinSize:
            thresh2[output == i + 1] = 255
    
#     plt.imshow(thresh2)
    thresh2 = np.uint8(thresh2)
    
    return thresh2

# Example:
# thresh2 = SpkltThresh(cropped=cropped_rgb, ResizeFactor=30, thr2=0.8, MinSize=1000)
# plt.imshow(thresh2)









# Spike's contours
def LabelContours(cropped_rgb, thresh2, ResizeFactor=30, MinSize = 1000, plot=True, thr2=0.8):
    
    # Copy iamge
    OutImage = cropped_rgb.copy()
    
    if thresh2 is not None:
        thresh2 = thresh2
    else:
        # Threshold for contours
        thresh2 = SpkltThresh(cropped=OutImage, ResizeFactor=ResizeFactor, thr2=thr2, MinSize=MinSize)

#     # Threshold for contours
#     thresh2 = SpkltThresh(cropped=cropped_rgb, ResizeFactor=30, thr2=thr2, MinSize=MinSize)
#     plt.imshow(bw_cont)
    
    # Enumerate objects
    # EnumerateSpkCV(thresh2, OutImage, TextSize=5, TROE2020=False)
    
#     contours = measure.find_contours(blurred,(0.8*255))
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     len(contours)
    
    # Detected spikelets
    print("Detected spikeletes: ", len(contours))
    
    if plot==True:
        img = OutImage.copy()
        # Plot all found contours
        plot_contours = cv2.drawContours(img, contours, -1, (0,255,0), 10)
        plt.imshow(plot_contours)
    
    return contours

# Example:
# labels_cont = LabelContours(cropped_rgb, thresh2=thresh2, ResizeFactor=30, MinSize = 1000, thr2=0.8, plot=True)
# len(labels_cont)









def ObjProps(labeled, cropped_rgb, cropped_lab, cropped_hsv, ImagePath, MinSize = 1000, rm_envelope=False):
    
    # Label + regionprops
    labeled_contours, num_contours = label(labeled, return_num = True)
    props_contours = regionprops(labeled_contours)
#     plt.imshow(labeled_contours)

    # # Create column with image name
    Image_Name = ImagePath.split('\\')[-1]
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
    H_props = regionprops(labeled_contours, intensity_image=cropped_hsv[:,:,0])
    S_props = regionprops(labeled_contours, intensity_image=cropped_hsv[:,:,1])
    V_props = regionprops(labeled_contours, intensity_image=cropped_hsv[:,:,2])
    
    red = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in red_props])
    green = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in green_props])
    blue = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in blue_props])
    L = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in L_props])
    a = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in a_props])
    b = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in b_props])
    H = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in H_props])
    S = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in S_props])
    V = np.array([[rp.mean_intensity,rp.min_intensity,rp.max_intensity] for rp in V_props])
    
    Red_Perc = np.array(channel_percentiles(red_props, Negatives=False)).T
    Green_Perc = np.array(channel_percentiles(green_props, Negatives=False)).T
    Blue_Perc = np.array(channel_percentiles(blue_props, Negatives=False)).T
    L_Perc = np.array(channel_percentiles(L_props)).T
    a_Perc = np.array(channel_percentiles(a_props, Negatives=True)).T
    b_Perc = np.array(channel_percentiles(b_props, Negatives=True)).T
    H_Perc = np.array(channel_percentiles(H_props)).T
    S_Perc = np.array(channel_percentiles(S_props)).T
    V_Perc = np.array(channel_percentiles(V_props)).T

    # Dataframe 1: for single obervation per spike
    Objects_per_image = pd.DataFrame(
    list(zip(Image_Name, Labels, Areas, MajorAxes, MinorAxes, Orientations, Eccentricities, Perimeters, 
             red[:,0], red[:,1], red[:,2], green[:,0], green[:,1], green[:,2], blue[:,0], blue[:,1], blue[:,2], 
             L[:,0], L[:,1], L[:,2], a[:,0], a[:,1], a[:,2], b[:,0], b[:,1], b[:,2], 
             H[:,0], H[:,1], H[:,2], S[:,0], S[:,1], S[:,2], V[:,0], V[:,1], V[:,2], 
             Red_Perc[:,0], Red_Perc[:,1], Red_Perc[:,2], Red_Perc[:,3], Red_Perc[:,4], Red_Perc[:,5], Red_Perc[:,6], 
             Green_Perc[:,0], Green_Perc[:,1], Green_Perc[:,2], Green_Perc[:,3], Green_Perc[:,4], Green_Perc[:,5], Green_Perc[:,6], 
             Blue_Perc[:,0], Blue_Perc[:,1], Blue_Perc[:,2], Blue_Perc[:,3], Blue_Perc[:,4], Blue_Perc[:,5], Blue_Perc[:,6], 
             L_Perc[:,0], L_Perc[:,1], L_Perc[:,2], L_Perc[:,3], L_Perc[:,4], L_Perc[:,5], L_Perc[:,6], 
             a_Perc[:,0], a_Perc[:,1], a_Perc[:,2], a_Perc[:,3], a_Perc[:,4], a_Perc[:,5], a_Perc[:,6], 
             a_Perc[:,7], a_Perc[:,8], a_Perc[:,9], a_Perc[:,10], a_Perc[:,11], a_Perc[:,12], a_Perc[:,13], 
             b_Perc[:,0], b_Perc[:,1], b_Perc[:,2], b_Perc[:,3], b_Perc[:,4], b_Perc[:,5], b_Perc[:,6], 
             b_Perc[:,7], b_Perc[:,8], b_Perc[:,9], b_Perc[:,10], b_Perc[:,11], b_Perc[:,12], b_Perc[:,13],
             H_Perc[:,0], H_Perc[:,1], H_Perc[:,2], H_Perc[:,3], H_Perc[:,4], H_Perc[:,5], H_Perc[:,6],
             S_Perc[:,0], S_Perc[:,1], S_Perc[:,2], S_Perc[:,3], S_Perc[:,4], S_Perc[:,5], S_Perc[:,6],
             V_Perc[:,0], V_Perc[:,1], V_Perc[:,2], V_Perc[:,3], V_Perc[:,4], V_Perc[:,5], V_Perc[:,6])), 
    columns = ['Image_Name', 'ObjLabel', 'Area', 'MajorAxis', 'MinorAxes', 'Orientation', 'Eccentricity', 'Perimeter', 
               'Red_mean', 'Red_min', 'Red_max', 'Green_mean', 'Green_min', 'Green_max', 'Blue_mean', 'Blue_min', 'Blue_max', 
               'L_mean', 'L_min', 'L_max', 'a_mean', 'a_min', 'a_max', 'b_mean', 'b_min', 'b_max',
               'H_mean', 'H_min', 'H_max', 'S_mean', 'S_min', 'S_max', 'V_mean', 'V_min', 'V_max', 
               'Red_p25', 'Red_p50', 'Red_p75', 'Red_Mean', 'Red_sd', 'Red_Min', 'Red_Max', 
               'Green_p25', 'Green_p50', 'Green_p75', 'Green_Mean', 'Green_sd', 'Green_Min', 'Green_Max', 
               'Blue_p25', 'Blue_p50', 'Blue_p75', 'Blue_Mean', 'Blue_sd', 'Blue_Min', 'Blue_Max', 
               'L_p25', 'L_p50', 'L_p75', 'L_Mean', 'L_sd', 'L_Min', 'L_Max', 
               'a_p25_pos', 'a_p50_pos', 'a_p75_pos', 'a_Mean_pos', 'a_sd_pos', 'a_Min_pos', 'a_Max_pos', 
               'a_p25_neg', 'a_p50_neg', 'a_p75_neg', 'a_Mean_neg', 'a_sd_neg', 'a_Min_neg', 'a_Max_neg', 
               'b_p25_pos', 'b_p50_pos', 'b_p75_pos', 'b_Mean_pos', 'b_sd_pos', 'b_Min_pos', 'b_Max_pos', 
               'b_p25_neg', 'b_p50_neg', 'b_p75_neg', 'b_Mean_neg', 'b_sd_neg', 'b_Min_neg', 'b_Max_neg',
               'H_p25', 'H_p50', 'H_p75', 'H_Mean', 'H_sd', 'H_Min', 'H_Max',
               'S_p25', 'S_p50', 'S_p75', 'S_Mean', 'S_sd', 'S_Min', 'S_Max',
               'V_p25', 'V_p50', 'V_p75', 'V_Mean', 'V_sd', 'V_Min', 'V_Max'])

    Objects_per_image['Circularity'] = (4 * np.pi * Objects_per_image['Area']) / (Objects_per_image['Perimeter'] ** 2)

    
    # Unique labels
    labels2 = np.unique(labeled_contours[labeled_contours > 0])

    # Empty list for contours
    C = []

    # Loop thorugh labels and add to list of contours
    for label2 in labels2:
            y = labeled_contours==label2
            y = y * 255
            y = y.astype('uint8')
            contours, hierarchy = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # len(contours)
            contours = np.squeeze(contours)
            C.append(contours)
        
    # List for angles
    Slopes = []
    
    contours = C    
    counter = 0
    
    for c in contours:
        # c = contours[0]
        counter = counter+1
        # print("working on contour ", counter)

        ellipse = cv2.fitEllipse(c)

        # Fit a line 
        rows,cols = cropped_rgb.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01);
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)

        rise = (0,lefty)[1] - (cols-1,righty)[1]
        run = cols
        Slope = rise/run
        Slopes.append(Slope)
    
    # Add slopes to data frame
    Objects_per_image['ObjAngle'] = Slopes
    
    # Remove first row, corresponding to spikes' envelope
    if rm_envelope==True:
        return Objects_per_image.iloc[1: , :]
    else:
        return Objects_per_image


# plt.imshow(spklts)
# Example
# Props = ObjProps(spklts, cropped_rgb, cropped_lab, ImagePath=img_name, MinSize = 5000)
# Props = ObjProps(labeled=thresh2, cropped_rgb=cropped_rgb, cropped_lab=cropped_lab, ImagePath=img_name, MinSize = 1000)
# WSProps = ObjProps(labeled=spklts, cropped_rgb=cropped_rgb, cropped_lab=cropped_lab, cropped_hsv=cropped_hsv, ImagePath=img_name, MinSize = 5000)









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

# Example
# enh0 = EnhanceImage(InputImage=cropped_rgb, Color = 3, Contrast = None, Sharp = 10)
# plt.imshow(enh0)









def LabelSpklts(cropped_rgb, MinDist=50, labels_out=True, n_spklt=True, ElliPlot=False, Plot=True):
    
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
    # plt.imshow(opening)

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
    coords = peak_local_max(distance, min_distance=MinDist, labels=bw_spklts)
    # plt.imshow(coords)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, spikelets = ndi.label(mask)
    
    # Watershed
    labels = watershed(-distance, markers, mask=bw_spklts)
#     plt.imshow(labels)

    labels2 = np.unique(labels[labels > 0])

    C = []

    for label in labels2:
            y = labels == label
            y = y * 255
            y = y.astype('uint8')
            contours, hierarchy = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # len(contours)
            contours = np.squeeze(contours)
            C.append(contours)

#         plt.imshow(d[2])
    
    contours = C

    if ElliPlot==True and Plot==False:
        
        OutImage = cropped_rgb.copy()

        # Plot all found contours
        OutImage = cv2.drawContours(OutImage, contours, -1, (0,0,0), 10);
        # plt.imshow(OutImage)

        for c in contours:
            # Generate random colors
            random_channels = (np.random.choice(range(256), size=3))
            rr = int(random_channels[0])
            rg = int(random_channels[1])
            rb = int(random_channels[2])
            
            ellipse = cv2.fitEllipse(c)
            OutImage = cv2.ellipse(OutImage,ellipse,(rr,rg,rb),10);

            # Fit a line 
            rows,cols = OutImage.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01);
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            # OutImage = cv2.line(OutImage,(cols-1,righty),(0,lefty),(rr,rg,rb),3);
            
            # Slope from tope left, which is is the origin [0,0]
            rise = (0,lefty)[1] - (cols-1,righty)[1]
            run = cols
            Slope = rise/run
            Slopes.append(Slope)            
        
        # Plot
        plt.imshow(OutImage)

    # Add slopes to data frame
    # Props['Spklt_Angle'] = Slopes

    if Plot==True and ElliPlot==False:  
        # Plot
        plt.imshow(labels, cmap=plt.cm.nipy_spectral)
        
        # Print number of spikelets detected
        print('Detected spikelets = ', spikelets)
    
    # Return labels
    if labels_out==True and n_spklt==True:
        return labels, spikelets

# Example:
# spklts, n_spklts= LabelSpklts(cropped_rgb, MinDist=50, labels_out=True, n_spklt=True, ElliPlot=True, Plot=False)
# plt.imshow(spklts==2)









def CountorProps(cropped_rgb, cropped_lab, cropped_hsv, thresh2, ImagePath, ResizeFactor=30, MinSize = 1000, thr2=0.8, plot=True, rm_envelope=False):
    
#     labels_cont = SpkContours(cropped_rgb, ResizeFactor=30, MinSize = 1000, thr2=0.8, plot=True)
    
    # Copy image
    OutImage = cropped_rgb.copy()
    # Get lab
    OutImageLab = color.rgb2lab(cropped_lab)
    
    if thresh2 is not None:
        thresh2 = thresh2
    else:
        # Threshold for contours
        thresh2 = SpkltThresh(cropped=OutImage, ResizeFactor=ResizeFactor, thr2=thr2, MinSize=MinSize)

    #     plt.imshow(thresh2)
    
    # Enumerate objects
#     EnumerateSpkCV(thresh2, OutImage, TextSize=3, TROE2020=False)
    
#     contours = measure.find_contours(blurred,(0.8*255))
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     len(contours)
    # np.array(contours).shape
    # Contour properties
    Props = ObjProps(labeled=thresh2, cropped_rgb=cropped_rgb, cropped_lab=cropped_lab, cropped_hsv=cropped_hsv, ImagePath=ImagePath, rm_envelope=False)
    
    # Detected spikelets
    # print("Fitted contours: ", len(contours))
    
    # Create list for slopes
    Slopes = []
#     len(Slopes)

    if plot==True:

        # Plot all found contours
        OutImage = cv2.drawContours(OutImage, contours, -1, (255,255,255), 10);

        for c in contours:
            # Generate random colors
            random_channels = (np.random.choice(range(256), size=3))
            rr = int(random_channels[0])
            rg = int(random_channels[1])
            rb = int(random_channels[2])
            
            ellipse = cv2.fitEllipse(c)
            OutImage = cv2.ellipse(OutImage,ellipse,(rr,rg,rb),10);

            # Fit a line 
            rows,cols = OutImage.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01);
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            OutImage = cv2.line(OutImage,(cols-1,righty),(0,lefty),(rr,rg,rb),3);
            
            # Slope from tope left, which is is the origin [0,0]
            rise = (0,lefty)[1] - (cols-1,righty)[1]
            run = cols
            Slope = rise/run
            Slopes.append(Slope)            
        
        # Plot
        plt.imshow(OutImage)
        
    else:
        
        for c in contours:
            
            ellipse = cv2.fitEllipse(c)
            
            # Fit a line 
            rows,cols = OutImage.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01);
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)

            rise = (0,lefty)[1] - (cols-1,righty)[1]
            run = cols
            Slope = rise/run
            Slopes.append(Slope)
    
    if len(Slopes) != len(Props):
        Props['Contour_Angle'] = [np.nan] * len(Props)
    else:
        # Add slopes to data frame
        Props['Contour_Angle'] = Slopes
    
    
    # Remove first row, corresponding to spikes' envelope
    if rm_envelope==True:
        return Props.iloc[1: , :]
    else:
        return Props



# Example:
# labels_cont = CountorProps(cropped_rgb, cropped_lab, thresh2, ImagePath=img_name, ResizeFactor=30, MinSize = 1000, thr2=0.8, plot=True)









def Heatmat(a, frames=30):
    a = np.array(a)
    a = a.reshape(len(a), 1)
    aT = a.T
    mat = np.multiply(a, aT)
#     mata.shape

    for frame in range(frames):
        if frame < 10:
            name = "./GIFS/img_0"+str(frame)+".png"
        elif frame < 100:
            name = "./GIFS/img_00"+str(frame)+".png"
        else:
            name = "./GIFS/img_"+str(frame)+".png"
        
        new_mat = np.log10( 1+(mat**(frame) ) )
        sns.heatmap(new_mat)
        plt.savefig(name)
        plt.close()

# Example:
# Heatmat(a=Areas, frames=30)









def makeGIF(filenames, duration = 0.25, out_name=None):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('./GIFS/GIF.gif', images, duration=duration)


# Example:
# filenames = glob.glob("./GIFS/" + '*.png', recursive=False)
# makeGIF(filenames, duration = 0.25, out_name=None)









def DistAll(bw, HeatMap=True, HeatMapOut=False, spike_length=None):
    
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(bw), 
                                                                               connectivity=8)
    # len(centroids[1:][:])
    img_center = centroids[0][:]
    c_points = centroids[1:][:]
    c_df = pd.DataFrame(c_points, columns=["x","y"])
    # c_df

    # https://stackoverflow.com/questions/57107729/how-to-compute-multiple-euclidean-distances-of-all-points-in-a-dataset

    # Consider points as tuples in a list
    data = [ (float(x),float(y)) for x,y in c_df[['x', 'y']].values ]

    # Create empty list for distances
    distances = []

    for point in data:

        # Compute the Euclidean distance between the current point and all others
        euc_dist = [math.sqrt((point[0]-x[0] )**2 + (point[1]-x[1])**2) for x in data]

        # Append to list
        distances.append(euc_dist)

    # Convert list to array
    D = np.array(distances)
    
    if spike_length != None:
        # Express it as a fraction from spike length
        D = D/spike_length
    else:
        D = D
    
    # Heatmap
    if HeatMap==True:
        sns.heatmap(D)
    
    if HeatMapOut==True:
        hm = sns.heatmap(D)
        return D, hm
    else:
        return D

# Example:
# D = DistAll(bw=thresh2, HeatMap=True, spike_length=SL)
# D, hm = DistAll(bw=thresh2, HeatMap=False, HeatMapOut=True, spike_length=SL)
# D = DistAll(bw=spklts, HeatMap=True, spike_length=SL)









# https://stackoverflow.com/questions/54832694/how-to-represent-a-binary-image-as-a-graph-with-the-axis-being-height-and-width

def imgraph(image, axis=0):
    # image = blurred
    # Theshold image if desired (only 2d images)
    maxindex = np.argmax(image[:,:], axis=axis)

    # Plot graph
    plt.plot(image.shape[axis] - maxindex)
    plt.show()








    
def ComparePlots(rows, cols, ListImages, fontsize=10):
    plots = rows * cols
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(plots):
        ax[i].imshow(ListImages[i], cmap='gray')
        Title = "Image " + str(i)
        ax[i].set_title(Title, fontsize=fontsize)
    fig.tight_layout()
    plt.show()
    
# Example:
# ComparePlots(3,1,[cropped_rgb, cropped_lab, cropped_hsv])  
    

    
    
    
    
    
    
    
def SeparateSpikes(ImagePath, Outfile = None):
    
    # Remove bakground
    I = RemoveBackground(ImagePath, OtsuScaling=0.25, rgb_out=True, gray_out=True, lab_out=False, hsv_out=False, bw_out=True)
    rgb0 = I[0]
#     # Convert to gray
#     gray0 = img0 @ [0.2126, 0.7152, 0.0722]
# #     gray0 = img0 @ [0.2126, 0.7152, 0.0722]

#     # Threshold
#     otsu = filters.threshold_otsu(gray0)
#     bw0 = gray0 > otsu
#     bw1 = morphology.remove_small_objects(bw0, min_size=1.5e-05 * gray0.shape[0] * gray0.shape[1])
    bw0 = I[1]
#     plt.imshow(bw1)
    # Label spikes
    labeled_spks, num_spikes = label(bw0>0, return_num = True)
    
    # Loop through spikes
    for spk in range(1,num_spikes):
        
        # Select current spike
        myspk = labeled_spks == spk

        # Crop spike
        slice_x, slice_y = ndimage.find_objects(myspk)[0]
        cropped_spk = myspk[slice_x, slice_y]
        cropped_rgb = rgb0[slice_x, slice_y]
        cropped_rgb = np.where(cropped_spk[..., None], cropped_rgb, 0)
        
        # Add 10 pixels to each border
        padded = np.pad(cropped_rgb, pad_width=[(10, 10),(10, 10),(0, 0)], mode='constant')
        
        # Save image 
        im = Image.fromarray(padded)
        
        if Outfile == None:
            
            Split_Path = ImagePath.split("\\")
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
        print("Saved image as: " + OutName)

# Example:
# SeparateSpikes(ImagePath=img_name)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    