# #!/usr/bin/env python

import SpykFunctions as SF
import time
import pandas as pd
import numpy as np
from skimage.measure import label
from datetime import datetime, date
import signal


# Functions
df = pd.DataFrame()

def spike_seg_batch(images_path, Tests, Segm_Type):
    Images = SF.ListImages(images_path, imgformat='.tif', recursive=False)
    df_tests = pd.DataFrame()

    for thresh in Tests:
        now = datetime.now().strftime("%H:%M:%S")
        print(now,"---testing theshold: ", thresh, "\n\n")
        df_test = segmented_spks(Images, Segm_Type, thresh)
        df_tests = pd.concat([df_tests, df_test])

    return df_tests


def segmented_spks(Images, Segm_Type, threshold):
    df = pd.DataFrame()

    for img_name in Images:
        df_img = pd.DataFrame()
        start_time = time.time() # Track time
        print("image ", img_name)

        if Segm_Type == "otsu":
            I = SF.spike_segm(img_name, rescale_rgb=0.5, channel_thresh=None, OtsuScaling=threshold, rgb_out=False,
                           gray_out=False, lab_out=False, hsv_out=False, bw_out=True)
        elif Segm_Type == "ct":
            I = SF.spike_segm(img_name, rescale_rgb=0.5, channel_thresh=[0,threshold], rgb_out=False,
                           gray_out=False, lab_out=False, hsv_out=False, bw_out=True)
        else:
            print("Define segmentation type as 'ct' or as 'otsu' ")
            break

        end_time = time.time()
        total_time = round(end_time - start_time, 2) # Total time

        bw0 = I[0]
        _, num_spikes = label(bw0, return_num = True)
        print('detected ', num_spikes)

        df_img['Segm_Type'] = [Segm_Type]
        df_img['Test'] = [threshold]
        df_img['Image_Name'] = [img_name]
        df_img['n_Spikes'] = [num_spikes]
        df_img['exec_time'] = [total_time]

        df = pd.concat([df,df_img])

    return df









# Tests
ct = [5,10,15,20,25,30,35,40,45,50,55,60]
otsu_factor = [0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35]

# images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/TEST/Test"

# images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/PGR_Roseau2021"

images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/PGR_GH2021"

ct_df = spike_seg_batch(images_path, ct, Segm_Type='ct')
ct_df.to_csv(images_path + "/ct_seg.csv")

otsu_df = spike_seg_batch(images_path, otsu_factor, Segm_Type='otsu')
otsu_df.to_csv(images_path + "/otsu_seg.csv")
