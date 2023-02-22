# #!/usr/bin/env python

# Spikelet Segmentation with variable Minimum Distance

import SpykFunctions as SF
import time
import pandas as pd
import numpy as np
from skimage.measure import label
from datetime import datetime, date
import signal
import scipy.ndimage as ndi
from pathlib import Path
import sys
from PIL import Image


# Functions

def Test_MinDist(Images, SpikeletFolder, Tests):
    df = pd.DataFrame()

    for img_name in Images:
        Image_Name = img_name.split('/')[-1]
        now = datetime.now().strftime("%H:%M:%S")
        print('\n\n')
        print(now +  " ---- Image ", Image_Name)

        I = SF.spike_segm(img_name, rescale_rgb=0.5, channel_thresh=[0,30], rgb_out=True, gray_out=False, lab_out=False, hsv_out=False, bw_out=True)

        rgb0 = I[0]
        bw0 = I[1]

        labeled_spks, num_spikes = label(bw0, return_num = True)

        for Label in range(1, num_spikes+1):
            now = datetime.now().strftime("%H:%M:%S")
            print(now +  " ------- Processing spike ", Label)
            spk = labeled_spks==Label
            # Crop spike
            slice_x, slice_y = ndi.find_objects(spk)[0]
            cropped_spk = spk[slice_x, slice_y]
            cropped_rgb = rgb0[slice_x, slice_y]
            cropped_rgb = np.where(cropped_spk[..., None], cropped_rgb, 0)

            # print("On to testing distances")

            for Dist in Tests:
                # signal.alarm(60)
                print("----Testing Label " + str(Label) + " MinDist " + str(Dist))

                try:
                    _,EllipseData,Spikelets_Image = SF.spikelet_segm(
                        cropped_rgb=cropped_rgb,Pad=200,MinDist=Dist,data_out=True,
                        plot_ellipse=True,Numbered=True,img_out=True,plot_segmented=False)

                    n_spikelets = EllipseData.shape[0]
                    df_spk = pd.DataFrame()
                    # print("Spikelet seg DONE\n\n\n")
                    df_spk['MinDist'] = [Dist]
                    df_spk['Image_Name'] = [Image_Name]
                    df_spk['Spike_Label'] = [Label]
                    df_spk['n_spikelets'] = [n_spikelets]
                    df = pd.concat([df,df_spk])
                    # df = pd.concat([df,EllipseData])
                    Filename = SpikeletFolder + "/md_" + str(Dist)
                    Path(Filename).mkdir(parents=True, exist_ok=True)
                    Filename = Filename + "/" + Image_Name.replace('.tif','')
                    Filename = Filename + '_'+ str(Label) + '.jpg'
                    im = Image.fromarray(Spikelets_Image)
                    im.save(Filename)
                    # print("Images saved as " + str(Filename))


                except:
                    print("\nThis labeled spike was ignored\n")
                    df_spk = pd.DataFrame()
                    df_spk['MinDist'] = [Dist]
                    df_spk['Image_Name'] = [Image_Name]
                    df_spk['Spike_Label'] = [Label]
                    df_spk['n_spikelets'] = [0]

                    df = pd.concat([df,df_spk])

                    # print(EllipseData.shape)
                    # print(df.n_spikelets)
                # signal.alarm(0)

    return df

# images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/TEST/Test"

# images_path = "J:\My Drive\M.Sc\THESIS\ImageAnalysis\SpikeProperties\SpykProps\Images\TEST\Test"

images_path = "/Volumes/GoogleDrive/My Drive/M.Sc/THESIS/ImageAnalysis/SpikeProperties/SpykProps/Images/PGR_Roseau2021"

Images = SF.ListImages(images_path, imgformat='.tif', recursive=False)

Tests = [25,30,35,40,45,50,55,60,65,70,75]
# Tests = [25,50,75]

date_time_now = datetime.now().strftime("%y%m%d_%H%M")
outfolder = images_path + "/" + date_time_now

df = Test_MinDist(Images, outfolder, Tests)
df.to_csv(outfolder + "/df_MinDist.csv")
