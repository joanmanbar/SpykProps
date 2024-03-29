# #!/usr/bin/env python

# Author: Joan Barreto Ortiz

# imports
import sys
import os
import argparse
import time
from datetime import datetime, date
# sys.path.append(".")
import SpykFunctions as SF
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage import color
from skimage.measure import label
import scipy.ndimage as ndi
import numpy as np
import cv2
import signal

# Parser
parser = argparse.ArgumentParser(description='Run SpykProps on an a folder or image path')
parser.add_argument('-d','--img_directory', type=str, required=True, help='Path to images folder or to single image')
parser.add_argument('-f','--img_format', type=str, default='.tif', help='(str) Images format')
parser.add_argument('-r','--rescale_rgb', type=float, default=None, help='(float) Rescale factor to resize original images')
parser.add_argument('-ct','--channel_thresh', help="(str) Channel and threshold values separated by a comma, respectively, for spike segmentation. Example: '-ct=0,30' threshold pixel values above 30 in channel 0.")
parser.add_argument('-md','--min_dist', type=int, default=50, help='(int) Minimum distance (in pixels) between spikelets. Suggested: 50 for original size; 40 for images rescaled at 0.75 or 0.5.')
parser.add_argument('-spklt','--spikelet_data', action='store_true', help='Returns spikelet properties')
parser.add_argument('-dist','--distances_data', action='store_true', help='Returns matrix of distances among spikelets')
parser.add_argument('-efd','--Fourier_desc', action='store_true', help='Returns Elliptical Fourier Descriptors (EFD) per spike.')
parser.add_argument('-nh','--n_harmonics', type=int, default=None, help='(int) Number of harmonics for EFD')
parser.add_argument('-timg','--track_image', action='store_true', help='Prints processing time for each image')
parser.add_argument('-tspk','--track_spike', action='store_true', help='Prints tracked spike')
parser.add_argument('-cc','--crop_coords', help="(str) Cropping coordinates for original RGB images. Must be separated by a comma, with the range for the Y axis (row numbers) being the first two values, and the following represent the the X axis (column numbers). Example: '-cc=44,6940,25,4970' takes only pixels from 44 to 6940 on the vertical axis, and 25 to 4970 on the horizontal axis.")

args = parser.parse_args()

# Define variables
InputPath = args.img_directory
if os.path.isdir(InputPath):
    Images = SF.ListImages(InputPath, imgformat=args.img_format, recursive=False)
if os.path.isfile(InputPath):
    Images = [InputPath]

if args.rescale_rgb != None:
    rescale_rgb = args.rescale_rgb
    resc_print = rescale_rgb
else:
    rescale_rgb = None
    resc_print = "None"

if args.channel_thresh != None:
    channel_thresh = args.channel_thresh.split(',')
    channel_thresh = list(map(int, channel_thresh))
    segm_print = channel_thresh
else:
    channel_thresh = None
    segm_print = "Otsu by 0.25"

if args.crop_coords != None:
    cc = args.crop_coords.split(',')
    cc = list(map(int, cc))

if str(args.spikelet_data) == "True":
    SpikeletData = True
else:
    SpikeletData = False

if str(args.distances_data) == "True":
    EucDist = True
else:
    EucDist = False

if str(args.Fourier_desc) == "True":
    EFD = True
    import spatial_efd
    import pyefd
    from pyefd import elliptic_fourier_descriptors
    from pyefd import normalize_efd
else:
    EFD = False

if args.n_harmonics != None:
    n_harmonics = args.n_harmonics
    nh_print = n_harmonics
    EFD_data = pd.DataFrame()
else:
    n_harmonics = None
    nh_print = "None"

if args.track_image != None:
    tck_img = True
else:
    tck_img = False

if args.track_spike == True:
    tck_spk = True
else:
    tck_spk = False

MinDist = args.min_dist

# This is to ignore skimage's warning on multichannel vs channel_axis
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Define output folders
date_time_now = datetime.now().strftime("%y%m%d_%H%M")
OutFolder = Images[0].rsplit('/', 1)[0]
OutFolder = OutFolder + '/' + date_time_now
Path(OutFolder).mkdir(parents=True, exist_ok=True)

# Create output folders
SpikeFolder = OutFolder + '/SpikeSegm/' # Spike segmentation
Path(SpikeFolder).mkdir(parents=True, exist_ok=True)
SpikeletFolder = OutFolder + '/SpikeletSegm/' # Spike segmentation
Path(SpikeletFolder).mkdir(parents=True, exist_ok=True)
LengthFolder = OutFolder + '/SpikeLength/' # Spike length approximation
Path(LengthFolder).mkdir(parents=True, exist_ok=True)
EFD_Folder = OutFolder + '/EFD/' # Spike length approximation
Path(EFD_Folder).mkdir(parents=True, exist_ok=True)
print('\n*********************************')
print('\nCreated folder (date_time)',OutFolder)

# Log file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        logFilename = OutFolder + "/logfile.txt"
        self.log = open(logFilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()


# Define function
def SpykBatch():

    print("\n*********************************\n\nSettings: ",
      "\n  Images' path: ", args.img_directory,
     "\n  Images's format: ", args.img_format,
     "\n  Rescale factor: ", resc_print,
     "\n  Method for spike segmentation: ", str(segm_print),
     "\n  Spikelets minimum distance: ", str(MinDist),
      "\n  Return spikelets dataset: ", str(args.spikelet_data),
      "\n  Return spikelets Euclidean distances: ", str(args.distances_data),
      "\n  Return EFD dataset: ", str(args.Fourier_desc),
      "\n  Number of harmonics for EFD: ", str(nh_print),
      "\n  Tracking time to process image: ", str(args.track_image),
      "\n  Tracking spikes: ", str(args.track_spike),
      "\n  Crooping coordinates: ", str(cc),
     "\n\n*********************************\n")

    print('Starting analysis...\n\n')

    start_time = time.time()
    Nimages = str(len(Images))
    Spikes_data = pd.DataFrame()
    Spklts_data = pd.DataFrame()
    Distances_data = pd.DataFrame()
    EFD_data = pd.DataFrame()
    Counter=0

    for img_name in Images:
        try:
            Image_Name = img_name.split('/')[-1]
            Counter = Counter+1

            if tck_img == True:
                now = datetime.now().strftime("%H:%M:%S")
                Progress = str(Counter) + "/" + Nimages
                print(now + " -- Processing image " + Progress + ": \n", img_name)
                # Set the initial time per image
                image_time = time.time()

            # Spike segmentation
            I = SF.spike_segm(img_name, rescale_rgb=rescale_rgb, channel_thresh=channel_thresh,
                           OtsuScaling=0.25, rgb_out=True, gray_out=True, lab_out=True,
                           hsv_out=True, bw_out=True,
                           crop_coord=cc)
            rgb0=I[0]; gray0=I[1]; lab0=I[2]; hsv0=I[3]; bw0 = I[4]

            # Enumerate spikes (to check spike segmentation)
            Enum_spike = SF.EnumerateSpkCV(bw=bw0, rgb=rgb0, TextSize=None,
                                      Plot=False, PlotOut=True)
            Filename = SpikeFolder + Image_Name.replace('.tif','.jpg')
            im = Image.fromarray(Enum_spike)
            im.save(Filename)

            # Collect spikes data (doesn't include spike length)
            df = SF.SpikesDF(I=I, ImagePath=img_name)

            if EucDist == True:
                # Matrix of distances among detected spikelets
                DistMat = pd.DataFrame(list(zip(Image_Name_List, Spk_Index, SpkDists)), columns = ['Image_Name', 'Spike_Label', 'MatrixD'])
                # Append
                # Distances_data = Distances_data.append(DistMat)
                Distances_data = pd.concat([Distances_data,DistMat])

            # Label spikes
            labeled_spks, num_spikes = label(bw0, return_num = True)

            # Prepare empty lists and DFs
            SpkLengths = []    # Spike lengths
            SpkDists = []    # spikelet distances
            DistancesPerSpk = pd.DataFrame()    # Spikelet distances data
            SpkltsPerSpk = pd.DataFrame()    # Spikelets data

            # Start at 1 so it ignores background (0)
            for Label in range(1, num_spikes+1):

                spk = labeled_spks==Label
                if tck_spk == True:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(now +  " ---- Processing spike ", Label)
                # Crop spike
                slice_x, slice_y = ndi.find_objects(spk)[0]
                cropped_spk = spk[slice_x, slice_y]
                cropped_rgb = rgb0[slice_x, slice_y]
                cropped_rgb = np.where(cropped_spk[..., None], cropped_rgb, 0)
                cropped_gray = color.rgb2gray(cropped_rgb)
                cropped_lab = color.rgb2lab(cropped_rgb)
                cropped_hsv = color.rgb2hsv(cropped_rgb)

                # Spike length
                sl, length_img = SF.spk_length(cropped_spk, Method='skel_ma',Overlay=True)
                SpkLengths.append(sl)
                Filename = LengthFolder + Image_Name.replace('.tif','.jpg')
                Filename = Filename + '_spk_'+ str(Label) + '.jpg'
                length_img.save(Filename)

                # EFD
                if EFD == True:
                    CoeffsPerSpike = pd.DataFrame()
                    Filename = EFD_Folder + Image_Name.replace('.tif','')
                    Filename = Filename + '_'+ str(Label)
                    coeff_df = SF.efd(cropped_spk, n_harmonics=30, plot_efd=True, efd_filename=Filename)
                    coeff_df['Image_Name'] = Image_Name
                    coeff_df['Spike_Label'] = Label
                    # coeffs['Min_Coeffs'] = harmonics
                    CoeffsPerSpike = pd.concat([CoeffsPerSpike,coeff_df])
                EFD_data = pd.concat([EFD_data,CoeffsPerSpike])

                # Spikelet segmentation
                try:

                    Spikelets,EllipseData,Spikelets_Image = SF.spikelet_segm(
                        cropped_rgb=cropped_rgb,Pad=200,MinDist=MinDist,data_out=True,
                        plot_ellipse=True,Numbered=True,img_out=True,plot_segmented=False)
                    Filename = SpikeletFolder + Image_Name.replace('.tif','')
                    Filename = Filename + '_'+ str(Label) + '.jpg'
                    im = Image.fromarray(Spikelets_Image)
                    im.save(Filename)

                    if SpikeletData==True:
                        SpikeletProps = SF.SpikeletsDF(labeled=Spikelets,Pad=200,cropped_rgb=cropped_rgb,
                                           cropped_lab=cropped_lab, cropped_hsv=cropped_hsv,
                                           ImagePath=img_name)
                        SpikeletProps = pd.concat([EllipseData,SpikeletProps], axis=1)

                        # Add spike label
                        SpikeletProps['Spike_Label'] = [str(Label)] * len(SpikeletProps)

                        SpkltsPerSpk = pd.concat([SpkltsPerSpk,SpikeletProps])

                except Exception as e:
                    print('Error processing spike:', Label)
                    print(e)
                    pass

            # Spklts_data = Spklts_data.append(SpkltsPerSpk)
            if SpikeletData==True and SpkltsPerSpk.empty == False:
                Spklts_data = pd.concat([Spklts_data,SpkltsPerSpk])

            # Add spike lengths and append current data frame
            df["SpykLength"] = SpkLengths
            # Spikes_data = Spikes_data.append(df)
            Spikes_data = pd.concat([Spikes_data,df])
            # Create columns with image name, spike index, and distances matrix
            Image_Name_List = [Image_Name] * len(SpkDists)
            Spk_Index = [number for number in range(1, num_spikes+1)]

            # How long did it take to run this image?
            if tck_img == True:
                print("Image " + img_name.split('\\')[-1] + ", \n" + Progress + ", \nwas fully processed in " + str(round(time.time() - image_time, 1)) + " seconds. " + "\n")

        except Exception as e:

            print('Error processing image: ', img_name)
            print(e)
            pass

    # How long did it take to run the whole code?
    print("\n\nRun time: ",
          str(round(time.time() - start_time, 1)), "seconds",
          "\nProcessed Images: ", len(Images),
          " \nProcessed spikes: ", len(Spikes_data),
          "\nAll data was saved in \n", OutFolder
         )

    # Need to reorder columns (geometric, then spectral)
    Spikes_data.to_csv(OutFolder + "/Spikes_data.csv", index=False)
    Spklts_data.to_csv(OutFolder + "/Spikelets_data.csv", index=False)
    Distances_data.to_csv(OutFolder + "/EucDistances_data.csv", index=False)
    EFD_data.to_csv(OutFolder + "/EFD_data.csv", index=False)





if __name__ == '__main__':
    SpykBatch()
