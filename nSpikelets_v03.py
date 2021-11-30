# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:55:08 2019

@author: jbarreto
"""




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









# # Gather the image files (change path)
# Images = io.ImageCollection(r'J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\Spyk_Prop\CSSA\Images\IN\*.tif')


# Define the function
def SpikeProps(Im):
    
    # Read image
    Im = Images.files[1]
    image0 = iio.imread(Im)
    # io.imshow(image0)
    
    # Crop image based on scanner's margins and pink tape
    image1 = image0[44:6940, 25:4970, :]
    # io.imshow(image1)
    image_rescaled = rescale(image1[...], 0.25, preserve_range=True, multichannel=True)
    final_image = image_rescaled.astype(np.uint8)
    image1 = final_image
#    io.imshow(image1)
    
    # Image type
    #print(type(img1))
    #print(img1.dtype)
    #print(img1.shape)
    #print(img1.min(), img1.max())
    
    # Assign each color channel to a different variable
    red = image1[:, :, 0]
    green = image1[:, :, 1]
    blue = image1[:, :, 2]
    
    # Threshold based on the red channel (this depends on the image's background)
    bw0 = red > 40
#    io.imshow(bw0)  
    ## Remove noise
    # bw1 = morphology.remove_small_objects(bw0, min_size=5000) # Filter out objects whose area < 10000
    bw1 = morphology.remove_small_objects(bw0, min_size=5000) # Filter out objects whose area < 10000
#    io.imshow(bw1)
    
    # Apply mask to RGB
    # image2 = np.asarray(image1)
    image2 = np.where(bw1[..., None], image1, 0)        # third condition changes mask's background on a 1-255 scale
#    io.imshow(image2)    
    
    # Convert to gray
    gray0 = image2 @ [0.2126, 0.7152, 0.0722]
    # io.imshow(gray0, cmap='gray')
    
    gray0_eq = exposure.equalize_hist(gray0)  # Give it a mask with borders!
    # io.imshow(gray0_eq, cmap='gray')
    
    
    # # Enhance gray to improve watershed
    # #https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    
    # img = gray0.astype(np.uint8)  # Converts to uint8
    
    # exposure.histogram(gray0)
    
    # # Contrast stretching
    # p2, p98 = np.percentile(img, (5, 95))
    # img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    # io.imshow(img_rescale)
    
    # # Equalization
    # img_eq = exposure.equalize_hist(img)
        
    
    # # Adaptive Equalization
    # img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    
    # # Display results
    # fig = plt.figure(figsize=(8, 5))
    # axes = np.zeros((2, 4), dtype=np.object)
    # axes[0, 0] = fig.add_subplot(2, 4, 1)
    # for i in range(1, 4):
    #     axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    # for i in range(0, 4):
    #     axes[1, i] = fig.add_subplot(2, 4, 5+i)
    
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
    # ax_img.set_title('Low contrast image')
    
    # y_min, y_max = ax_hist.get_ylim()
    # ax_hist.set_ylabel('Number of pixels')
    # ax_hist.set_yticks(np.linspace(0, y_max, 5))
    
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    # ax_img.set_title('Contrast stretching')
    
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    # ax_img.set_title('Histogram equalization')
    
    # ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    # ax_img.set_title('Adaptive equalization')
    
    # ax_cdf.set_ylabel('Fraction of total intensity')
    # ax_cdf.set_yticks(np.linspace(0, 1, 5))
    
    # # prevent overlap of y-axis labels
    # fig.tight_layout()
    # plt.show()














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
    # centr_x, centr_y = props['centroid']
    
    if not os.path.exists('.\labeled'):
        os.makedirs('.\labeled')
    
    fig, ax = plt.subplots()
    ax.imshow(bw1, cmap=plt.cm.gray)
    
    spike_ind = 0
    
    for props in regions:
        y0, x0 = props.centroid
        spike_ind = spike_ind + 1
        plt.text(x0, y0, str(spike_ind), color="red", fontsize=20)
  
    
    plt.show()
    pylab.savefig('.\labeled\foo.png')
    
    
    # Get labels
    # labels = measure.label(bw1)
    
    # Plot region props of objects in image
    fig = px.imshow(bw1, binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info
    
    props = regionprops(labeled_spks, bw1)
    properties = ['area', 'eccentricity', 'perimeter', 'mean_intensity']
    centroid = ['centroid']
    
    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labeled_spks.max()):
        label = props[index].label
        contour = measure.find_contours(labeled_spks == label, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))
    
    fig


    
    # Iterate all particles, add label and diameters to input image
    for i, p in enumerate(particles):
        x = p[0]
        y = max(0, p[1]-10)
        d_h = p[2] / scale[0][2] * 500
        d_v = p[3] / scale[0][2] * 500
        cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        print('Particle ' + str(i) + ' | Horizontal diameter: ' + '{:.2f}'.format(d_h) +
              ' nm, vertical diameter: ' +  '{:.2f}'.format(d_v) + ' nm')
    
    cv2.imshow('img', cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5))
    cv2.imshow('thresh', cv2.resize(thresh, dsize=(0, 0), fx=0.5, fy=0.5))
    cv2.imshow('img_mop',  cv2.resize(img_mop, dsize=(0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    # Mean colors
    red_props = regionprops(labeled_spks, intensity_image=red)
    red_means = [rp.mean_intensity for rp in red_props]
    
    green_props = regionprops(labeled_spks, intensity_image=green)
    green_means = [gp.mean_intensity for gp in green_props]
    
    blue_props = regionprops(labeled_spks, intensity_image=blue)
    blue_means = [bp.mean_intensity for bp in blue_props]
    
    # Determine regions properties
    regions = regionprops(labeled_spks)
#    io.imshow(labeled_spks == 1)
    
    
    # ------------- BRANCHES -------------
# From: https://jni.github.io/skan/install.html
    
    # Create a completely empty dataframe for branches in spikes in image
    Branches_per_image = pd.DataFrame()
    
    # Create a completely empty dataframe for spikelets in spike in image
    Spikelets_per_image = pd.DataFrame()
    
    
    # Lenght tests
    # Lengths_test = []
    
    # Loop through each spike in image    
    for spk in range(1, num_spikes):
        
        
        # io.imshow(labeled_spks)  
#       io.imshow(labeled_spks == 3)
        spk = 3
        myspk = labeled_spks == spk
        
        
        # Rotate spike (for improved watershed)
        # Rotate spike
        # Adapted from: https://python-forum.io/Thread-finding-angle-between-three-points-on-a-2d-graph
        # Get orientation
        current_angle = regions[spk-1].orientation
        
        # Get Centroid coordinates
        Cx = regions[spk-1].centroid[1]
        Cy = regions[spk-1].centroid[0]
        
        # Get point A's coordinates
        Ax = Cx - math.cos(current_angle) * 0.5 * regions[spk-1].major_axis_length
        Ay = Cy + math.sin(current_angle) * 0.5 * regions[spk-1].major_axis_length
        
        # Get point B's coordinates
        Bx = myspk.shape[0]
        By = Cy
        
        # Calculate rotation angle
        rot_ang = math.degrees(math.atan2(Bx-Cx, By-Cy) - math.atan2(Ax-Cx, Ay-Cy))
        
        myspk_rot = rotate(myspk, angle=rot_ang, resize = True)
        io.imshow(myspk_rot)


        # Rotate spike
        # Adapted from: https://python-forum.io/Thread-finding-angle-between-three-points-on-a-2d-graph
        # fig, ax = plt.subplots()
        # ax.imshow(myspk, cmap=plt.cm.gray)
        
        # ax.plot((Cx, Ax), (Cy, Ay), '-r', linewidth=1.5)
        # ax.plot((Cx, Bx), (Cy, Cy), '-r', linewidth=1.5)
        # ax.plot(Cx, Cy, '.g', markersize=10)
        
        # plt.show()

        
        # minr, minc, maxr, maxc = regions[spk-1].bbox
        # total_rows = maxr - minr
        # total_cols = maxc - minc
        
        # if total_rows > total_cols:
        #     bbox_orientation = 'Vertical'
        #     if c_ang < 0:
        #         # Rotation angle
        #         rot_ang = -c_ang + 90
        # else:
        #     bbox_orientation = 'Horizontal'
        
        # if c_ang > 0:
        #     # Rotation angle
        #     rot_ang = -c_ang
        # else:
        #     rot_ang = np.pi * (c_ang)
      
        
    
        
        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
        # ax0.imshow(testing)
        # ax1.imshow(myspk)
        
        
        
        
        # MORPHOLOGYCAL OPERATIOS
        
        # Structural element
        # se0 = morphology.disk(5)    # use diamond(30) if RGB is at 100% size
        # io.imshow(se0)
        
        # Dilation
        # dilated = morphology.closing(myspk, selem=se0, out=None)
        # io.imshow(dilated)
        
        # # Blurry with Gaussian filter
        # blurry = filters.gaussian(dilated, sigma = 15)
        # # io.imshow(blurry)
        # myspk0 = blurry > 0.5
        # # io.imshow(myspk0)
        
        


        # SKELETONS      
        
        # https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html#sphx-glr-auto-examples-edges-plot-skeleton-py
        # Medial axis
        # skel = morphology.medial_axis(myspk)
        skel = morphology.skeletonize(myspk)
        # io.imshow(skel)
        # skel, distance = morphology.medial_axis(dilated, return_distance=True)
        # Compare with other skeletonization algorithm
        # skeleton = morphology.skeletonize(dilated)
        # spike_length = np.count_nonzero(spike_length_skel==1)
        # io.imshow(skeleton)
        
        # Distance to the background for pixels of the skeleton
        # dist_on_skel = distance * skel

        # fig, axes = plt.subplots(1, 2, figsize=(8, 8), sharex=True, sharey=True)
        # ax = axes.ravel()
                
        # ax[0].imshow(skeleton, cmap='magma')
        # ax[0].contour(dilated, [0.5], colors='w')
        # ax[0].set_title('skeletonize')
        # ax[0].axis('off')
        
        # ax[1].imshow(skel, cmap='magma')
        # ax[1].contour(dilated, [0.5], colors='w')
        # ax[1].set_title('medial_axis')
        # ax[1].axis('off')
        
        # fig.tight_layout()
        # plt.show()
        
        
        
        # Dilation on skel
        # se1 = morphology.diamond(5)
        # dil_skel = morphology.dilation(skel, selem=se1, out=None, shift_x=False, shift_y=False)
        # io.imshow(dil_skel)       
        
        # Fill holes
        skel1 = scipy.ndimage.morphology.binary_fill_holes(skel)
#        io.imshow(skel1)
        
        # Skeletonize
        skel1 = morphology.medial_axis(skel1)
#        io.imshow(skel1)
        
        # Verify there is no more holes
        # skel2 = scipy.ndimage.morphology.binary_fill_holes(skel1)
#        io.imshow(skel2)
        
        # Skeletonize
        # skel2 = morphology.skeletonize(skel2)
#        io.imshow(skel2)
        
        # skel2 = morphology.thin(skel1)
      
        
        
        # Plot skeleton on binary
        fig, ax = plt.subplots()
        draw.overlay_skeleton_2d(myspk, skel1, dilate=2, axes=ax);        
        
    
        
        
        
        
        
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
























