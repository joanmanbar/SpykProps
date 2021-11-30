# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:49:26 2020

@author: jbarreto
"""

# Function to improve spike image

import numpy as np

from skimage import io
from skimage import morphology
from skimage import filters
from skimage.measure import label
# from skimage.exposure import histogram
# from skimage.filters import sobel

from scipy import ndimage
# import scipy.misc

# import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance 




# input parameters
# mydir = r"C:\Users\jbarreto\Pictures\EpsonV600\Panicles_E5_Fall2019"
# random_pic = 69
# random_spyk = 1
# out_dir = 'J:\My Drive\PROJECTS\TURFGRASS\PRG\Spikes\Python\out_files'


def ImproveSpyk(IN_folder = 0, OUT_folder = 0, Pic_Num = 0, Spyk_Num = 0, Col = 0, Sharp = 0, ImpSpk_name = 0):
    
    
    if len(str(IN_folder)) > 1:
        mydir = IN_folder
        mydir = str(mydir) + "\\*.tif"
        mydir = io.ImageCollection(mydir)
    else:
        mydir = input("Enter input folder's path to images. \n")
        mydir = str(mydir) + "\\*.tif"
        mydir = io.ImageCollection(mydir)
    
    if len(str(OUT_folder)) > 1:
        outdir = OUT_folder
    else:
        outdir = input(input("Enter output folder's path to images. \n"))
    
    
    if len(str(Pic_Num)) > 1:
        random_pic = int(Pic_Num)
    else:
        random_pic = int(input("Select a number between 1 and " + str(len(mydir)) + "\n"))
    
    
    if len(str(Spyk_Num)) > 1:
        random_spyk = int(Spyk_Num)
    else:
        random_spyk = int(input("Select a spike number. \n"))
    
    
    # random_pic = int(input("Select a number between 1 and " + str(len(mydir)) + "\n"))
    
    # random_spyk = int(input("Select a spike number."))
    # random_spyk = int(input())
    
    # out_dir = input(input("Enter output folder's path to images. \n"))
    
    img_name = mydir.files[random_pic]
    # img_name = [mydir + '\\' + mypic + '.tif']
    
    
    # Read image
    img0 = io.imread(img_name)
    
    # Crop image based on scanner's margins and pink tape
    img0 = img0[44:6940, 25:4970, :]
    #    io.imshow(image1)
    
    # Convert to gray
    gray0 = img0 @ [0.2126, 0.7152, 0.0722]
    
    # Set image threshold
    T = filters.threshold_otsu(gray0)
    
    # Segment gray image
    bw0 = gray0 > T 
    
    bw1 = morphology.remove_small_objects(bw0, min_size=1000) # Filter out objects whose area < 1000
    
    # Label spikes
    labeled_spks, num_spikes = label(bw1, return_num = True)
    
    # isolate a spike
    myspk0 = labeled_spks == random_spyk
    
    # Crop bw image
    slice_x, slice_y = ndimage.find_objects(myspk0)[0]
    c_myspk0 = myspk0[slice_x, slice_y]
    
    # crop RGB
    c_RGB0 = img0[slice_x, slice_y]
    c_RGB0 = np.asarray(c_RGB0)
    c_RGB0 = np.where(c_myspk0[...,None], c_RGB0, 0)
    
    # Add border to bw
    # myspk1 = np.pad(c_myspk0, pad_width=10, mode='constant', constant_values=0)
    
    # Add border to RGB
    myspkRGB0 = np.stack([np.pad(c_RGB0[:,:,c], pad_width=10, mode='constant', constant_values=0) for c in range(3)], axis=2)
    
    
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
    # ax0.imshow(myspk1)
    # ax1.imshow(myspkRGB0)
    
    # plt.imshow(myspkRGB0)
    
    
    
    
    # RGB enhancement
    img0 = Image.fromarray(myspkRGB0)
    img1 = ImageEnhance.Color(img0)
    
    if len(str(Col)) > 1:
        img1 = img1.enhance(Col)
    else:
        img1 = img1.enhance(3.5)
    
    # Sharpness (Good ~20 or higher)
    img2 = ImageEnhance.Sharpness(img1)
    
    if len(str(Sharp)) > 1:
        img2 = img2.enhance(Sharp)
    else:
        img2 = img2.enhance(20)
    
    # Final image
    img2 = np.asarray(img2)
    
    
    Out_collection = outdir + "\\*.jpg"
    Out_collection = io.ImageCollection(Out_collection)
    
    # filename
    if len(str(ImpSpk_name)) > 1:
        out_image = OUT_folder + '\\' + ImpSpk_name + ".jpg"
    else:
        # Out_Pic_Num = str(len(outdir) + 1)
        img_name = img_name.split('\\')[-1]
        Img_Num = len(Out_collection)+1
        img_name = img_name + "_0" + str(Img_Num) + ".jpg"
        out_image = outdir + '\\' + img_name
    
    
    im = Image.fromarray(img2)
    im.save(out_image)
    
    print("\n \n \nImage was saved as: \n" + img_name + "\nin: " + outdir + "\n \n \n")



















