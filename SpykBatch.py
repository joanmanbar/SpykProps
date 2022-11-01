# #!/usr/bin/env python

# Author: Joan Barreto Ortiz

# imports
import sys
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


# Parser
parser = argparse.ArgumentParser(description='Run SpykProps on an a folder')
parser.add_argument('-d','--img_directory', type=str, metavar='', required=True, help='Folder containing images to process')
parser.add_argument('-f','--img_format', type=str, metavar='', default='.tif', help='(str) Images format')
parser.add_argument('-r','--rescale_rgb', type=float, metavar='', default=None, help='(float) Rescale factor to resize original images')
parser.add_argument('-ct','--channel_thresh', metavar='', help="(str) Channel and threshold values separated by a comma, respectively, for spike segmentation. Example: '-ct=0,30' threshold pixel values above 30 in channel 0.")
parser.add_argument('-md','--min_dist', type=int, metavar='', default=50, help='(int) Minimum distance (in pixels) between spikelets. Suggested: 50 for original size; 40 for images rescaled at 0.75 or 0.5.')
parser.add_argument('-qc','--quality_control', type=bool, metavar='', default=True, help='(bool) Quality control. If True, creates folders with images of segmented spikes, segmented spikelets, and spike length detection.')
parser.add_argument('-spklt','--spikelet_data', type=bool, metavar='', default=True, help='(bool) Return spikelet properties')
parser.add_argument('-dist','--distances_data', type=bool, metavar='', default=False, help='(bool) Return matrix of distances among spikelets')
parser.add_argument('-efd','--Fourier_desc', type=bool, metavar='', default=False, help='(bool) Return Elliptical Fourier Descriptors (EFD) per spike.')
parser.add_argument('-nh','--n_harmonics', type=int, metavar='', default=None, help='(int) Number of harmonics for EFD')


args = parser.parse_args()



# Define variables
Images = SF.ListImages(args.img_directory, imgformat=args.img_format, recursive=False)

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
    
if args.spikelet_data == False:
    SpikeletData = False
    spklt_df_rpint = "False"
else:
    SpikeletData = True
    spklt_df_rpint = "True"
    
if args.distances_data == False:
    EucDist = False
    dist_df_rpint = "False"
else:
    EucDist = True
    dist_df_rpint = "True"
    
if args.Fourier_desc == False:
    EFD = False
    efd_print = "False"
else:
    EFD = True
    efd_print = "True"
    import spatial_efd
    import pyefd
    from pyefd import elliptic_fourier_descriptors
    from pyefd import normalize_efd
    
if args.n_harmonics != None:
    n_harmonics = args.n_harmonics
    nh_print = n_harmonics
    
    EDF_cols = ['An','Bn','Cn','Dn']
    EDF_nums = list(range(1,n_harmonics+1))
    EDF_nums = [str(x) for x in EDF_nums]
    Cols = np.char.add(EDF_cols,np.array(EDF_nums)[:,None])
    Cols = Cols.reshape(1,4*n_harmonics)
    Cols = Cols.tolist()
    Cols = Cols[0]
    EFD_data = pd.DataFrame()
else:
    n_harmonics = None
    nh_print = "None"
    
MinDist = args.min_dist
QC = args.quality_control


# Define output folders

date_time_now = datetime.now().strftime("%y%m%d_%H%M")  
OutFolder = Images[0].rsplit('/', 1)[0]
OutFolder = OutFolder + '/' + date_time_now
if QC == True:
    SpikeFolder = OutFolder + '/SpikeSegm/' # Spike segmentation 
    Path(SpikeFolder).mkdir(parents=True, exist_ok=True)     
    SpikeletFolder = OutFolder + '/SpikeletSegm/' # Spike segmentation 
    Path(SpikeletFolder).mkdir(parents=True, exist_ok=True)      
    LengthFolder = OutFolder + '/SpikeLength/' # Spike length approximation 
    Path(LengthFolder).mkdir(parents=True, exist_ok=True) 
    print('\n*********************************')
    print('\nCreated folder (date_time)',OutFolder)
    

# Define function    
def SpykBatch():

    print("\n*********************************\n\nSettings: ",
      "\n  Images' folder: ", args.img_directory,
     "\n  Images's format: ", args.img_format,
     "\n  Rescale factor: ", resc_print,
     "\n  Method for spike segmentation: ", str(segm_print),
     "\n  Spikelets minimum distance: ", str(MinDist),
      "\n  Return spikelets dataset: ", str(spklt_df_rpint),
      "\n  Return spikelets Euclidean distances: ", str(dist_df_rpint),
      "\n  Return EFD dataset: ", str(efd_print),
      "\n  Number of harmonics for EFD", str(nh_print),
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
            Progress = str(Counter) + "/" + Nimages
            # img_name=Images[0]
            now = datetime.now().strftime("%H:%M:%S")
            print(now + " -- Processing image " + Progress + ": \n", img_name)
            # img_name = Images[9]
            # Set the initial time per image
            image_time = time.time()

            # Spike segmentation
            I = SF.spike_segm(img_name, rescale_rgb=rescale_rgb, channel_thresh=channel_thresh, 
                           OtsuScaling=0.25, rgb_out=True, gray_out=True, lab_out=True, 
                           hsv_out=True, bw_out=True, crop_coord=[44,6940,25,4970])
            rgb0 = I[0]
            gray0 = I[1]
            lab0 = I[2]
            hsv0 = I[3]
            bw0 = I[4]    

            # Enumerate spikes (to check spike segmentation)
            if QC == True:
                Enum_spike = SF.EnumerateSpkCV(bw=bw0, rgb=rgb0, TextSize=None, 
                                          Plot=False, PlotOut=True)
                Filename = SpikeFolder + Image_Name.replace('.tif','.jpg')
                im = Image.fromarray(Enum_spike)
                im.save(Filename)

            # Collect spikes data (doesn't include spike length)
            df = SF.SpikesDF(I=I, ImagePath=img_name)

            # Label spikes
            labeled_spks, num_spikes = label(bw0, return_num = True)

            # Prepare empty lists and DFs
            SpkLengths = []    # Spike lengths
            SpkDists = []    # spikelet distances 
            SpkltsPerSpk = pd.DataFrame()    # Spikelets data
            DistancesPerSpk = pd.DataFrame()    # Spikelet distances data     

            # Start at 1 so it ignores background (0) 
            for Label in range(1, num_spikes+1):
                try:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(now +  " ---- Processing spike ", Label)
                    spk = labeled_spks==Label
                    # Crop spike
                    slice_x, slice_y = ndi.find_objects(spk)[0]
                    cropped_spk = spk[slice_x, slice_y]
                    cropped_rgb = rgb0[slice_x, slice_y]
                    cropped_rgb = np.where(cropped_spk[..., None], cropped_rgb, 0)
                    cropped_gray = color.rgb2gray(cropped_rgb)
                    cropped_lab = color.rgb2lab(cropped_rgb)
                    cropped_hsv = color.rgb2hsv(cropped_rgb)

                    # Spike length 
                    if QC == True:
                        sl, length_img = SF.spk_length(cropped_spk, Method='skel_ma',Overlay=True)
                        SpkLengths.append(sl)
                        Filename = LengthFolder + Image_Name.replace('.tif','.jpg')
                        Filename = Filename + '_spk_'+ str(Label) + '.jpg'
                        length_img.save(Filename)   

                    else:
                        sl, length_img = SF.spk_length(cropped_spk, Method='skel_ma',Overlay=True)
                        SpkLengths.append(sl)

                    # Spikelet segmentation
                    if QC == True:
                        Spikelets,EllipseData,Spikelets_Image = SF.spikelet_segm(
                            cropped_rgb=cropped_rgb,Pad=200,MinDist=MinDist,data_out=True,
                            plot_ellipse=True,Numbered=True,img_out=True,plot_segmented=False)
                        Filename = SpikeletFolder + Image_Name.replace('.tif','')
                        Filename = Filename + '_'+ str(Label) + '.jpg'
                        im = Image.fromarray(Spikelets_Image)
                        im.save(Filename)
                    else:
                        Spikelets,EllipseData = SF.spikelet_segm(
                            cropped_rgb=cropped_rgb,Pad=200,MinDist=MinDist,data_out=True,
                            plot_ellipse=False,Numbered=False,img_out=False,plot_segmented=False)

                    if SpikeletData==True:
                        SpikeletProps = SF.SpikeletsDF(labeled=Spikelets,Pad=200,cropped_rgb=cropped_rgb, 
                                           cropped_lab=cropped_lab, cropped_hsv=cropped_hsv,
                                           ImagePath=img_name)
                        SpikeletProps = pd.concat([EllipseData,SpikeletProps], axis=1)

                    # Distances between spikelets (ellipses' centroids)
                    if EucDist == True:
                        D = SF.DistAll(EllipseData=EllipseData, HeatMap=False, spike_length=sl)
                        SpkDists.append(D)     
                        
                    if EFD == True:
                        CoeffsPerSpike = pd.DataFrame()
                        bw_efd = np.pad(cropped_spk, pad_width=[(100, 100),(100, 100)], mode='constant')
                        bw_efd = ndi.binary_fill_holes(bw_efd)
                        bw_efd = np.uint8(bw_efd)
                        contour, hierarchy = cv2.findContours(bw_efd,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                        contour = np.squeeze(contour)
                        x = contour[:,0]
                        y = contour[:,1]
                        # First n coefficients
                        coeffs = spatial_efd.CalculateEFD(x,y,n_harmonics)
                        coeffs, rotation = spatial_efd.normalize_efd(coeffs, size_invariant=True)
                        coeffs = coeffs.reshape(1,4*n_harmonics)

                        # Min number of coefficients
                        # Nyquist Freqency: maximum number of harmonics that can be computed for a given contour
                        nyquist = spatial_efd.Nyquist(x)
                        tmpcoeffs = spatial_efd.CalculateEFD(x, y, nyquist+1)
                        harmonics = spatial_efd.FourierPower(tmpcoeffs, x)
                        # coeffs_min = spatial_efd.CalculateEFD(x, y, harmonics)
                        # coeffs_min, rotation = spatial_efd.normalize_efd(coeffs_min, size_invariant=True)

                        coeffs = pd.DataFrame(coeffs, columns=Cols)
                        coeffs['Image_Name'] = Image_Name
                        coeffs['Spike_Label'] = Label
                        coeffs['Min_Coeffs'] = harmonics
                        CoeffsPerSpike = pd.concat([CoeffsPerSpike,coeffs])

                    # Add spike label
                    SpikeletProps['Spike_Label'] = [str(Label)] * len(SpikeletProps)

                    EFD_data = pd.concat([EFD_data,CoeffsPerSpike])
                    SpkltsPerSpk = pd.concat([SpkltsPerSpk,SpikeletProps])

                except Exception as e:
                    print('There was an error with this spike:')
                    Spikes_data.to_csv(OutFolder + "/Spikes_data.csv", index=False)
                    Spklts_data.to_csv(OutFolder + "/Spikelets_data.csv", index=False)
                    Distances_data.to_csv(OutFolder + "/EucDistances_data.csv", index=False)
                    EFD_data.to_csv(OutFolder + "/EFD_data.csv", index=False)
                    print(e)
                    pass

            # Spklts_data = Spklts_data.append(SpkltsPerSpk)
            Spklts_data = pd.concat([Spklts_data,SpkltsPerSpk])
            # Distances_data = Distances_data.append(SpkDists)

            # Add spike lengths and append current data frame
            df["SpykLength"] = SpkLengths
            # Spikes_data = Spikes_data.append(df)
            Spikes_data = pd.concat([Spikes_data,df])

            # Create columns with image name, spike index, and distances matrix
            Image_Name = [Image_Name] * len(SpkDists)
            Spk_Index = [number for number in range(1, num_spikes+1)]

            if EucDist == True:
                # Matrix of distances among detected spikelets
                DistMat = pd.DataFrame(list(zip(Image_Name, Spk_Index, SpkDists)), columns = ['Image_Name', 'Spike_Label', 'MatrixD'])
                # Append
                # Distances_data = Distances_data.append(DistMat)
                Distances_data = pd.concat([Distances_data,DistMat])

            # How long did it take to run this image?
            print("Image " + img_name.split('\\')[-1] + ", \n" + Progress + ", \nwas fully processed in " + str(round(time.time() - image_time, 1)) + " seconds. " + "\n")

        except Exception as e:
            print('there was an error!')
            Spikes_data.to_csv(OutFolder + "/Spikes_data.csv", index=False)
            Spklts_data.to_csv(OutFolder + "/Spikelets_data.csv", index=False)
            Distances_data.to_csv(OutFolder + "/EucDistances_data.csv", index=False)
            EFD_data.to_csv(OutFolder + "/EFD_data.csv", index=False)
            # Save current progress
            print("\n\nRun time: ", 
                  str(round(time.time() - start_time, 1)), "seconds", 
                  " \nProcessed spikes: ", len(Spikes_data), 
                  "\nAll data was saved in \n", OutFolder)
            print(e)
            pass

    # How long did it take to rusn the whole code?
    print("\n\nRun time: ", 
          str(round(time.time() - start_time, 1)), "seconds", 
          "\nProcessed Images: ", len(Images),
          " \nProcessed spikes: ", len(Spikes_data), 
          "\nAll data was saved in \n", OutFolder
         )

    # Reorder columns (geometric, then spectral)
    Spikes_data.to_csv(OutFolder + "/Spikes_data.csv", index=False)
    Spklts_data.to_csv(OutFolder + "/Spikelets_data.csv", index=False)
    Distances_data.to_csv(OutFolder + "/EucDistances_data.csv", index=False)
    EFD_data.to_csv(OutFolder + "/EFD_data.csv", index=False)
        
    


    
if __name__ == '__main__':
    SpykBatch()
