#!/usr/bin/env python3


#-----------------------------------------------------#
#                   SpykFunctions
#-----------------------------------------------------#
#
# This .py file contains the definitions to the functions to run SpykProps
#
#
#




import sys
sys.path.append(".")


# Dependencies
import sys
import glob
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import math
import pathlib
from pathlib import Path

from skimage import measure, segmentation, color, filters, morphology, color, feature, io, feature, util, morphology, exposure, img_as_float
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops, perimeter, find_contours
# from skimage.future import graph
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
from PIL import Image, ImageEnhance, ImageCms
# from skan import Skeleton, summarize, skeleton_to_csgraph, draw
import seaborn as sns
# import imutils
import imageio
import random
from datetime import datetime, date
import time








# list images
def ListImages(path, imgformat=".tif", recursive=False):
    Images = glob.glob(path + '/*' + imgformat, recursive=True)
    Images = [x.replace('\\', '/') for x in Images]
    return sorted(Images)

# Example:
# path = r'./Images/TEST'
# Images = ListImages(path, imgformat=".tif", recursive=False)









# Spike segmentation
def spike_segm(img, rescale_rgb=None, channel_thresh=None, OtsuScaling=0.25, 
               rgb_out=True, gray_out=True,lab_out=True, hsv_out=True, 
               bw_out=True, crop_coord=None):
    # Read image
    if isinstance(img, str) == True:
        img0 = io.imread(img)
    else:
        img0 = img 
    
    # Crop images with list (L) such that L=[top,bottom,left,right]
    # [44:6940, 25:4970, :]
    if crop_coord != None:
        img1 = img0[crop_coord[0]:crop_coord[1], 
                    crop_coord[2]:crop_coord[3], :]
    else:
        img1 = img0
        
    # Rescale
    if rescale_rgb != None:
        rescaled_rgb = rescale(img1[...], rescale_rgb, preserve_range=False, 
                       channel_axis=2, anti_aliasing=False)
        rescaled_rgb = 255 * rescaled_rgb
        rescaled_rgb = rescaled_rgb.astype(np.uint8)
        img1 = rescaled_rgb
        
    # Segmentation based on channel_thresh list (L) such that L = [channel, threshold in uint8]
    if channel_thresh != None:
        channel = channel_thresh[0]
        threshold = channel_thresh[1]
        bw0 = img1[:,:,channel] > threshold
    # Otsu segmentation (default)
    else: 
        # Convert to gray
        gray0 = img1 @ [0.2126, 0.7152, 0.0722]
        # Set image threshold
        T = filters.threshold_otsu(gray0)
        T = T*OtsuScaling
        # Segment gray image
        bw0 = gray0 > T
    # Remove small objects
    n_pixels = img1.shape[0] * img1.shape[1]
    minimum_size = n_pixels/10000
    bw1 = morphology.remove_small_objects(bw0, min_size=np.floor(minimum_size))
    
    # Masks
    ImagesOut = []
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
        ImagesOut.append(bw1)
    return ImagesOut

## Usage:
# %%time
# I = spike_segm(Images[3], rescale_rgb=None, channel_thresh=None, OtsuScaling=0.25, rgb_out=True, 
#                gray_out=True, lab_out=True, hsv_out=True, bw_out=True,
#               crop_coord=[44,6940,25,4970])
# rgb0 = I[0]
# gray0 = I[1]
# lab0 = I[2]
# hsv0 = I[3]
# bw0 = I[4]










# Enumerate spikes
def EnumerateSpkCV(bw, rgb, TextSize=None, Plot=True, PlotOut=False):
    
    if TextSize == None:
        TextSize = round(0.001 * np.sqrt(bw.shape[0] * bw.shape[1]), 1)
    else:
        TextSize = TextSize

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(bw), connectivity=8)
    img = rgb.copy()

    # ignore the background (centroid 0)
    centroids = centroids[1:]
    counter=1

    for c in centroids:
#         print(c)
        cx = round(c[0])
        cy = round(c[1])
        img = cv2.circle(img, (cx, cy), 10, (255, 0, 0), -1)
        img = cv2.putText(img, str(counter), (cx - 25, cy - 25),cv2.FONT_HERSHEY_SIMPLEX,
                          TextSize, (255, 0, 0), 2)
        counter = counter+1

    if Plot==True:
        plt.imshow(img)

    if PlotOut==True:
        return img

# # Example:
# EnumerateSpkCV(bw0, rgb0, TextSize=None, Plot=True, PlotOut=False)
# Enum_img = EnumerateSpkCV(bw0, rgb0, TextSize=5, Plot=False, PlotOut=True)
# EnumerateSpkCV(spklts, cropped_rgb, TextSize=5)








# Spike length
def spk_length(cropped_spk,Method='skel_ma',Overlay=True):

    cropped_spk = np.pad(cropped_spk, pad_width=[(100, 100),(100, 100)], mode='constant')
    
    Length_data = []
    Time_data = []
    
    # Bounding box 
    if Method == "all" or Method == "bbox":
        track_time = time.time() # Track time
        label_bbox = label(cropped_spk)
        bbox_MajorAxis_L = regionprops(label_bbox)
        bbox_MajorAxis_L  = round(bbox_MajorAxis_L[0]['major_axis_length'], 0)
        total_time = round(time.time() - track_time, 2) # Total time
        Length_data.append(bbox_MajorAxis_L)
        Time_data.append(total_time)
    
    # Convex hull
    if Method == "all" or Method == "chull":
        track_time = time.time() # Track time
        chull = convex_hull_image(cropped_spk)
        label_chull = label(chull)
        chull_MajorAxis_L = regionprops(label_chull)
        chull_MajorAxis_L  = round(chull_MajorAxis_L[0]['major_axis_length'], 0)
        total_time = round(time.time() - track_time, 2) # Total time        
        Length_data.append(chull_MajorAxis_L)
        Time_data.append(total_time)
    
    # Skeleton (Lee, 94)
    if Method == "all" or Method == "skel_ma":     
        track_time = time.time() # Track time
        # Severly blur the image
        blur = cv2.blur(np.float32(cropped_spk),(100,100))
        # Threshold the blur
        thrb = blur > 0.1
        skeleton_ma = medial_axis(thrb)
        # Spike length
        SpkL_ma = cv2.countNonZero(np.float32(skeleton_ma))
        total_time = round(time.time() - track_time, 2) # Total time        
        Length_data.append(SpkL_ma)
        Time_data.append(total_time)
        
    # Skeleton
    if Method == "all" or Method == "skel":     
        track_time = time.time() # Track time
        # Severly blur the image
        blur = cv2.blur(np.float32(cropped_spk),(100,100))
        # Threshold the blur
        thrb = blur > 0.1
        skeleton = skeletonize(thrb)
        # Spike length
        SpkL = cv2.countNonZero(np.float32(skeleton))
        total_time = round(time.time() - track_time, 2) # Total time        
        Length_data.append(SpkL)
        Time_data.append(total_time)

        
    # Visualize overlay?
    if Overlay == True and Method == 'skel_ma':
        # Dilate skeleton and color it
        dil_skel = ndimage.binary_dilation(skeleton_ma, np.ones((7,7)))
        rgb_skel = np.zeros((dil_skel.shape[0],dil_skel.shape[1],3), dtype=np.uint8)
        # Make True pixels red
        rgb_skel[dil_skel]  = [255,0,0]
        # Make False pixels blue
        rgb_skel[~dil_skel] = [0,0,0]
        # plt.imshow(rgb_skel)

        img_color = np.dstack((cropped_spk, cropped_spk, cropped_spk))
        img_hsv = color.rgb2hsv(img_color)
        color_mask_hsv = color.rgb2hsv(rgb_skel)
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1]
        img_masked = color.hsv2rgb(img_hsv)
        img_masked = Image.fromarray((img_masked*255).astype(np.uint8))
        
        return Length_data[0], img_masked
        
    elif Overlay == False and Method == "all":
        Length_data = pd.DataFrame(Length_data).transpose()
        Length_data = Length_data.rename(
            columns={0:'bbox',1:'chull',2:'skel_Lee',3:'skel'})
        
        Time_data = pd.DataFrame(Time_data).transpose()
        Time_data = Time_data.rename(
            columns={0:'bbox',1:'chull',2:'skel_Lee',3:'skel'})
        
        return Length_data, Time_data
    else:
        return Length_data[0]

## Example:
# SL, length_img = spk_length(cropped_spk, Overlay=True)
# Lengths,Time = spk_length(cropped_spk, Method='all', Overlay=False) 









def LengthBatch(Images):
    
    import time
    from datetime import datetime, date
    start_time = time.time()    
    Nimages = str(len(Images)) 
    Spikes_data = pd.DataFrame()
    Contours_data = pd.DataFrame()
    Spklts_data = pd.DataFrame()
    Distances_data = pd.DataFrame()   
    
    Lengths_data = pd.DataFrame()
    Duration_data = pd.DataFrame()
    
    Counter=0

    for img_name in Images:
        
        Counter = Counter+1
        Progress = str(Counter) + "/" + Nimages
        # img_name=Images[0]
        now = datetime.now().strftime("%H:%M:%S")
        print("Processing image " + Progress + ": ", img_name + " at " + now)
        # img_name = Images[9]

        # Set the initial time per image
        image_time = time.time()

        # Remove background and create images
        I = RemoveBackground(img_name, OtsuScaling=0.25, rgb_out=False, 
                 gray_out=False, lab_out=False, hsv_out=False, bw_out=True)
        bw0 = I[0]
        labeled_spks, num_spikes = label(bw0, return_num = True)

        SpkLengths = pd.DataFrame()
        TrackTime = pd.DataFrame()

        # Start at 1 so it ignores background (0) 
        for Label in range(1, num_spikes+1):

            spk = labeled_spks==Label
            # io.imshow(spk)

            # Crop spike
            slice_x, slice_y = ndimage.find_objects(spk)[0]
            cropped_spk = spk[slice_x, slice_y]

            # Spike length
            sl, t = spk_length(cropped_spk, Method='all', Overlay=False)
            SpkLengths = pd.concat([SpkLengths,sl], axis=0)  
            TrackTime = pd.concat([TrackTime,t], axis=0) 
        
        Image_Name = img_name.split('\\')[-1]
        Image_Name = [Image_Name] * num_spikes
        SpkLengths['Image_Name'] = Image_Name
        TrackTime['Image_Name'] = Image_Name
        
        Spike_Label = [number for number in range(1, num_spikes+1)]
        SpkLengths['Spike_Label'] = Spike_Label   
        TrackTime['Spike_Label'] = Spike_Label 
        
        Lengths_data = pd.concat([Lengths_data,SpkLengths], axis=0)
        Duration_data = pd.concat([Duration_data,TrackTime], axis=0)

        # How long did it take to run this image?
        print("Image " + img_name.split('\\')[-1] + ", " + Progress + ", was fully processed in " + str(round(time.time() - image_time, 1)) + " seconds. " + "\n")

    # How long did it take to run the whole code?
    print("This entire code took", str(round(time.time() - start_time, 1)), "seconds to run.")

    return Lengths_data, Duration_data

# Lengths, Time = LengthBatch(Images)









def PixelHist(bw, ColorSpace, channel = 0, spikes="All", nbins = 100):
    
    # Higher number of bins inreases computational power

    labeled_spks, num_spikes = label(bw, return_num = True)
#     plt.imshow(labeled_spks==0)

    if spikes=="All":
        labeled_spks = labeled_spks
    else:
        for L in range(num_spikes):
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
        # Colors.append(list(np.random.choice(range(2), size=3)))

    plt.hist(Spikes_Data, bins = nbins, color = Colors2, label = Names);

    # Plot formatting
    plt.legend();
    plt.xlabel('Intensity Value');
    plt.ylabel('Number of NonZero Pixels');
    plt.title('Distribution of None-Zero Pixel Values for Selected Given Channel and Spikes');

# Example:
# PixelHist(bw=bw0, ColorSpace=hsv0, channel = 2, spikes=[0,1,2], nbins = 25)
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
        # min_pos = min(positive_values)
        min_pos = np.nanpercentile(positive_values, 5)
        min_pos_list.append(min_pos)
        # max_pos = max(positive_values)
        max_pos = np.nanpercentile(positive_values, 95)
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
            # min_neg = min(negative_values)
            min_neg = np.nanpercentile(negative_values, 5)
            min_neg_list.append(min_neg)
            # max_neg = max(negative_values)
            max_neg = np.nanpercentile(negative_values, 95)
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









def SpikesDF(I, ImagePath):

    # list of images
    rgb0 = I[0]; gray0 = I[1]; lab0 = I[2]; hsv0 = I[3]; bw0 = I[4]
    # Label binary mask
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

    # Dataframe with single row per spike
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
    
    # Add Circularity and reorder df
    Spikes_per_image['Circularity'] = (4 * np.pi * Spikes_per_image['Area']) / (Spikes_per_image['Perimeter'] ** 2)
    # Reorder columns (geometric, then spectral)
    Cols = list(Spikes_per_image)
    Cols.insert(8, Cols.pop(Cols.index('Circularity')))
    Spikes_per_image = Spikes_per_image.loc[:, Cols]
    
    # Output
    return Spikes_per_image


# Example:
# df = SpikesDF(I, ImagePath=img_name)









def SpkltThresh(cropped, ResizeFactor=30, thr2=0.8, MinSize=1000, Save=False, Filename=None):

    # Check that it's a gray image
    if len(cropped.shape) > 2:
        # Convert to gray
        cropped_gray = color.rgb2gray(cropped)
    else:
        cropped_gray = cropped

    # cropped_gray = np.pad(cropped_gray, pad_width=[(100, 100),(100, 100)], mode='constant')

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

    kernel = morphology.disk(3)
    eroded = cv2.erode(thresh2, kernel, iterations=2)
    # plt.imshow(eroded, cmap='gray')

    se = morphology.disk(10)
    mask = ndimage.binary_opening(eroded, se)
    # plt.imshow(mask, cmap='gray')

    thresh2 = eroded


    if Save==True and Filename!=None:
        plt.imshow(thresh2)
        plt.savefig(Filename)
        plt.close()

    return thresh2

# # Example:
# thresh2 = SpkltThresh(cropped=cropped_rgb, ResizeFactor=30, thr2=0.8, MinSize=1000, Save=True, Filename="testing.png")
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









def SpikeletsDF(labeled,Pad,cropped_rgb,cropped_lab,cropped_hsv,ImagePath):
    
    # Was the image padded in spikelet_segm?
    if Pad!=None:
        labeled = labeled[Pad:-Pad, Pad:-Pad]        
        
    # Label + regionprops
    labeled_contours, num_contours = label(labeled, return_num = True)
    props_contours = regionprops(labeled_contours)

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
    # Reorder columns (geometric, then spectral)
    Cols = list(Objects_per_image)
    Cols.insert(8, Cols.pop(Cols.index('Circularity')))
    Objects_per_image = Objects_per_image.loc[:, Cols]
    
    # Output
    return Objects_per_image



# # Example
# Props = SpikeletsDF(Spikelets,Pad=200,cropped_rgb=cropped_rgb, cropped_lab=cropped_lab,
#                  cropped_hsv=cropped_hsv,ImagePath=img_name)









def EnhanceImage(InputImage, Color = None, Contrast = None, Sharp = None):

    # Read image
    if isinstance(InputImage, str) == True:
        img = imageio.imread(InputImage)
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









def spikelet_segm(cropped_rgb,Pad=200,MinDist=50,data_out=True,plot_ellipse=False,
                Numbered=False,img_out=False,plot_segmented=False):

    # Add pad - improves segmentation
    padded_rgb = np.pad(cropped_rgb, pad_width=[(Pad, Pad),(Pad, Pad),(0, 0)], mode='constant')
    # Rescale to 10% of original
    rescaled_spk = rescale(padded_rgb[...], 0.1, preserve_range=False, channel_axis=2, anti_aliasing=True)
    # Erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    erosion = cv2.erode(rescaled_spk,kernel,iterations = 1)
    # Opening
    kernel = np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations = 10)
    # Resize to original
    rescaled_spk2 = Image.fromarray((rescaled_spk * 255).astype(np.uint8))
    rescaled_spk2 = rescaled_spk2.resize((padded_rgb.shape[1],padded_rgb.shape[0]))
    # Opening
    opening = np.asarray(rescaled_spk2)
    # Convert rgb to gray
    gray_spklts = opening @ [0.2126, 0.7152, 0.0722]
    # Binarize gray (threshold may vary according to spike and background colors)
    bw_spklts = gray_spklts > 50
    # Get distances
    distance = ndi.distance_transform_edt(bw_spklts)
    # plt.imshow(-distance)
    # Get max peaks
    coords = peak_local_max(distance, min_distance=MinDist, labels=bw_spklts)
    # Create mask
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    # Label
    markers, spikelets = ndi.label(mask)
    # Watershed
    labels = watershed(-distance, markers, mask=bw_spklts)
    labels2 = np.unique(labels[labels > 0])
    # spikes shouldn't have more than 50 spikelets
    
    
    if len(labels2) < 2 or len(labels2) > 50:
        print("Error! Check spike segmentation!")
        return

    # Contours for segmented spikelets (C)
    C = []
    for Label in labels2:
        y = labels == Label
        y = y * 255
        y = y.astype('uint8')
        contours, hierarchy = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # len(contours)
        contours = np.squeeze(contours)
        C.append(contours)
    #         plt.imshow(y)

    contours = C

    # Export angles, areas, and centroids. Centroids are useful to estiamte distances between spikelets
    if data_out==True:
        
        EllipsesIndex = []
        EllipsesAngle = []
        EllipsesLength = []
        EllipsesArea = []
        EllipsesCentroid = []

        if plot_ellipse==True:
            OutImage = padded_rgb.copy()
            # Plot all found contours
            # OutImage = cv2.drawContours(OutImage, contours, -1, (0,0,0), 10);

        ellipse_ind = 0
        for c in contours:
            ellipse_ind = ellipse_ind + 1
            # Generate random colors

            random_channels = (np.random.choice(range(50,256), size=3))
            rr = int(random_channels[0])
            rg = int(random_channels[1])
            rb = int(random_channels[2])

            # Fit elipse and line
            # Source: https://stackoverflow.com/questions/62698756/opencv-calculating-orientation-angle-of-major-and-minor-axis-of-ellipse
            ellipse = cv2.fitEllipse(c)
            (xc,yc),(d1,d2),angle = ellipse
            rmajor = max(d1,d2)/2

            if angle > 90:
                angle = angle - 90
            else:
                angle = angle + 90

            xtop = xc + math.cos(math.radians(angle))*rmajor
            ytop = yc + math.sin(math.radians(angle))*rmajor
            xbot = xc + math.cos(math.radians(angle+180))*rmajor
            ybot = yc + math.sin(math.radians(angle+180))*rmajor

            ellipse_area = np.pi * d1 * d2
            ellipse_length = int(math.dist([int(xtop),int(ytop)],[int(xbot),int(ybot)]))
            EllipsesIndex.append(ellipse_ind)
            EllipsesAngle.append(angle)  
            EllipsesLength.append(ellipse_length)
            EllipsesArea.append(ellipse_area)
            EllipsesCentroid.append([int(xc),int(yc)])

            if plot_ellipse==True:
                OutImage = cv2.ellipse(OutImage,ellipse,(rr,rg,rb),3);
                cv2.line(OutImage, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (rr,rg,rb), 3)

            if Numbered == True:
                cv2.putText(OutImage, str(ellipse_ind), (int(xc) - 25, int(yc) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (rr,rg,rb), 5)

        # Dataframe for data
        EllipsesData = pd.DataFrame({'Ellipse_Number':EllipsesIndex,
                                 'Ellipse_Angle':EllipsesAngle,
                                 'Ellipse_Area':EllipsesArea,
                                     'Ellipse_Length':EllipsesLength,
                                    'Ellipse_Centroid':EllipsesCentroid})        
        if img_out==True:
            return(labels, EllipsesData, OutImage)
        else:
            return(labels, EllipsesData)

    if plot_segmented==True and plot_ellipse==False:  
        plt.imshow(labels, cmap=plt.cm.nipy_spectral)
        print('Detected spikelets = ', spikelets)
    
# # Example:
# Spikelets, EllipseData, Spikelets_Image  = spikelet_segm(
#     cropped_rgb=rgb0,Pad=200,MinDist=50,data_out=True,
#     plot_ellipse=True,Numbered=True,img_out=True,plot_segmented=False)









def CountorProps(cropped_rgb, cropped_lab, cropped_hsv,
                 thresh2, ImagePath, ResizeFactor=30,
                 MinSize = 1000, thr2=0.8, Plot=False,
                 Save=False):

    # Copy image
    OutImage = cropped_rgb.copy()
    # Get lab
    OutImageLab = color.rgb2lab(cropped_lab)

    if thresh2 is not None:
        thresh2 = thresh2
    else:
        # Threshold for contours
        thresh2 = SpkltThresh(cropped=OutImage, ResizeFactor=ResizeFactor, thr2=thr2, MinSize=MinSize)
        # Remove padding
        thresh2 = thresh2[100:-100, 100:-100]

    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    Props = ObjProps(labeled=thresh2, cropped_rgb=cropped_rgb, cropped_lab=cropped_lab,
                     cropped_hsv=cropped_hsv, ImagePath=ImagePath, rm_envelope=False)

    Props.drop(Props[Props.Area < 50].index, inplace=True)
    # q = Props["Area"].quantile(0.5)

    # Create list for slopes
    Slopes = []
#     len(Slopes)

    if Plot==False:

        for c in contours:

            if len(c)<5:
                continue

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

        return Props


    if Plot==True:

        # Plot all found contours
        OutImage = cv2.drawContours(OutImage, contours, -1, (0,255,255), 5);
        # plt.imshow(OutImage)

        # Enumerate objects
        OutImage = EnumerateSpkCV(thresh2, OutImage, TextSize=2, TROE2020=False, Plot=False, PlotOut=True)
        # plt.imshow(OutImage)

        for c in contours:

            if len(c)<5:
                continue

            # # Generate random colors
            # random_channels = (np.random.choice(range(256), size=3))
            # rr = int(random_channels[0])
            # rg = int(random_channels[1])
            # rb = int(random_channels[2])

            ellipse = cv2.fitEllipse(c)
            # OutImage = cv2.ellipse(OutImage,ellipse,(rr,rg,rb),10);
            OutImage = cv2.ellipse(OutImage,ellipse,(240,0,0),5);

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

        if len(Slopes) != len(Props):
            Props['Contour_Angle'] = [np.nan] * len(Props)
        else:
            # Add slopes to data frame
            Props['Contour_Angle'] = Slopes


        if Save==True:
            return Props, OutImage
        else:
            plt.imshow(OutImage)
            return Props



# # # Example:
# labels_cont, OutImage = CountorProps(cropped_rgb, cropped_lab, cropped_hsv,
#                            thresh2=thresh2, ImagePath=img_name,
#                            ResizeFactor=30, MinSize = 1000, thr2=0.8,
#                            plot=True, Save=True)










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








# Distances among all spikelets
def DistAll(EllipseData, HeatMap=True, HeatMapOut=False, spike_length=None):

    # Extract centroids from 'spikelet_segm' data
    c_points =list(EllipseData['Ellipse_Centroid'][:])
    c_df = pd.DataFrame(c_points, columns=["x","y"], 
                        index=EllipseData['Ellipse_Number'])
    
    # https://stackoverflow.com/questions/57107729/how-to-compute-multiple-euclidean-distances-of-all-points-in-a-dataset

    # Points as tuples in a list
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
        
    D = pd.DataFrame(D, index = EllipseData['Ellipse_Number'],
                  columns = EllipseData['Ellipse_Number'])
    # Heatmap
    if HeatMap==True:
        sns.heatmap(D)

    if HeatMapOut==True:
        hm = sns.heatmap(D)
        return D, hm
    else:
        return D

# # Example:
# D = DistAll(EllipseData=EllipseData, HeatMap=True)
# D, hm = DistAll(EllipseData=EllipseData, H HeatMap=False, HeatMapOut=True, spike_length=SL)










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




















###***********************************************************###
            # Functions for Machine Learning Purposes
###***********************************************************###







def read_and_Resize(image_path, ResizeFactor = 8, mask=False):

    if mask==True:
        labeled_img = plt.imread(image_path)
        labeled_img = labeled_img.astype(np.uint8)
        labeled_img = labeled_img[:,:,0]
        # plt.imshow(labeled_img)
        kernel = np.ones((3,3), np.uint8)
        labeled_img = cv2.erode(labeled_img, kernel, iterations=1)
        im = Image.fromarray((labeled_img).astype(np.uint8))
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im = Image.fromarray((img).astype(np.uint8))

    # Resize image
    (width, height) = (im.width // ResizeFactor, im.height // ResizeFactor)
    resized_img = im.resize((width, height))
    resized_img = np.asarray(resized_img)

    return(resized_img)

# img = read_and_Resize(image_path, ResizeFactor = ResizeFactor, mask=False)










def features_to_df(img, df_original):
    img2 = img.reshape(-1)
    df = df_original.copy()

    #Generate Gabor features
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5


                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label


    ########################################
    #Gerate OTHER FEATURES and add them to the data frame

    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    # plt.imshow(edges)
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe

    return(df)

# df2 = features_to_df(img, df_original=df)
# df = df2









def CollectTrainData(image_path, mask_path, ResizeFactor=8):

    # Create empty df
    df = pd.DataFrame()

    img = read_and_Resize(image_path, ResizeFactor = ResizeFactor, mask=False)

    img2 = img.reshape(-1)

    df['Original Image'] = img2

    df = features_to_df(img, df_original=df)

    labeled, unlabeled = add_labels(mask_path=mask_path, MinSize=30, MaxSize=200)

    #Remember that you can load an image with partial labels
    #But, drop the rows with unlabeled data
    df['Labels'] = labeled.reshape(-1)
    df['ToDrop'] = unlabeled.reshape(-1)

    df = df.drop(df[df['ToDrop'] == 1].index)
    df = df.drop(labels = ["ToDrop"], axis=1)
    Image_Name =  image_path.split('\\')[-1]
    df['Image_Name'] = [Image_Name] * len(df['Labels'])

    return(df)

# images_path = r'.\CVATLabeling\TROE2021_p003-p012\images\default'
# Images = ListImages(images_path, imgformat=".tif", recursive=False)

# masks_path = r'.\CVATLabeling\TROE2021_p003-p012\masks'
# Masks = ListImages(masks_path, imgformat=".tif", recursive=False)

# ResizeFactor = 8
# df_all = pd.DataFrame()

# for i in range(0,len(Images)):
#     df = CollectTrainData(image_path=Images[i],mask_path=Masks[i], ResizeFactor=8)
#     df_all = df_all.append(df)

# # Wall time: 1min 33s
# # About 13 sec per image/label









