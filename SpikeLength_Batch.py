# #!/usr/bin/env python

import SpykFunctions as SF
import time
import pandas as pd
import numpy as np
from skimage.measure import label
from datetime import datetime, date
import signal
import scipy.ndimage as ndi


# Functions

def length_methods(Images):
    df = pd.DataFrame()

    for img_name in Images:
        now = datetime.now().strftime("%H:%M:%S")
        print(now +  "\n\n---- Image ", img_name)

        I = SF.spike_segm(img_name, rescale_rgb=0.5, channel_thresh=[0,30], rgb_out=False, gray_out=False, lab_out=False, hsv_out=False, bw_out=True)

        bw0 = I[0]
        labeled_spks, num_spikes = label(bw0, return_num = True)

        for Label in range(1, num_spikes+1):
            # Set signal alarm for spike
            signal.alarm(10)
            now = datetime.now().strftime("%H:%M:%S")
            print(now +  " ------- Processing spike ", Label)
            spk = labeled_spks==Label
            # Crop spike
            slice_x, slice_y = ndi.find_objects(spk)[0]
            cropped_spk = spk[slice_x, slice_y]
            Lengths,Time = SF.spk_length(cropped_spk, Method='all', Overlay=False)
            df_spk = pd.concat([Lengths, Time], axis=1)
            df_spk['Image_Name'] = [img_name]
            df_spk['Spike_Label'] = [Label]

            df = pd.concat([df,df_spk])

    return df


# Tests
# ct = [5,10,15,20,25,30,35,40,45,50,55,60]
# otsu_factor = [0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35]

# images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/TEST/Test"

# images_path = "J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\SpykProps\Images\TEST\Test"

images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/PGR_Roseau2021"

Images = SF.ListImages(images_path, imgformat='.tif', recursive=False)

df = length_methods(Images)
df.to_csv(images_path + "/df_lengths.csv")



images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/PGR_GH2021"

Images = SF.ListImages(images_path, imgformat='.tif', recursive=False)

df = length_methods(Images)
df.to_csv(images_path + "/df_lengths.csv")
