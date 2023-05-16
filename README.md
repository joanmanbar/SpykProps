# SpykProps
(Requires Python 3.5 or newer version).

### Author:
Joan Bareto Ortiz (jbarreto@umn.edu)

This repository contains the Python code to collect spike (inflorescences) properties from images of perennial ryegrass (_Lolium perenne_ L.). The code was written for images collected with a flatbed scanner at 600 dpi using a black velvet as background. Differences in image size, resolution, or background can affect the output of the analysis. Please refer to the Overview.ipynb file to use SpykProps on single images using Jupyter-lab, or follow the instructions below for *batch processing*.

## Instructions for batch processing
These instructions

### 0. Clone repository
```
git clone https://github.com/joanmanbar/SpykProps.git
cd SpykProps
```

### 1. Add the images folder to the `SpykProps` repository
For example, if the folder is called 'MyImages', the structure for the current directory should look as follows:

```bash
SpykProps/
|-- MyImages/
|   |-- img_01.tif
|   |-- ...
|   |-- img_n.tif
|-- __init__.py
|-- SpykBatch.py
|-- SpykFunctions.py
|-- requirements.txt  
|-- README.md
|-- Overview.ipnb
```

### 2. Install requirements
```
pip install -r requirements.txt
```
*Note*: If a module (e.g. 'mymodule') is missing, install it as follows:

```
pip install mymodule
```

### 3. Batch function
This function will execute the program from command line on a given set of images and return the desired output. Check the help menu.
```
python SpykBatch.py -h
```

#### 3.1. Parameters

- `--img_directory` (`-d`): a string indicating the path to the images.
- `--img_format` (`-f`): a string indicating the images' format (`default='.tif'`).
- `--rescale_rgb` (`-r`): a float indicating a rescale factor to resize original images (`default=None`).
- `--channel_thresh` (`-ct`): a string containing the channel and threshold values for spike segmentation, respectively separated by a comma (`default=None`). If `None` is passed, segmentation will be based on the Otsu algorithm scaled by 0.25.
- `--min_dist` (`-md`): an integer indicating the minimum distance (in pixels) between spikelets (`default=50`). Suggested: 50 for original size; 40 for images rescaled at 0.75 or 0.5, depending on image size.
- `--spikelet_data` (`-spkl`): a boolean asking if user wants a dataset with spikelet properties (`default=True`).
- `--distances_data` (`-dist`): a boolean asking if user wants a dataset with euclidean distances between spikelets (`default=True`).
- `--Fourier_desc` (`-efd`): a boolean asking if user wants a dataset with elliptical Fourier coefficients per spikes (`default=True`).
- `--n_harmonics` (`-nh`): an integer indicating the number of harmonics for the elliptical Fourier coefficients (`default=None`).
- `--track_image`  (`-timg`): Prints processing time for each image.
- `--track_spike` (`-tspk`):   Prints tracked spike.
- `--crop_coords` (`-cc`): a string with cropping coordinates for original RGB images. Must be separated by a comma, with the range for the Y axis (row numbers) being the first two values, and the following represent the the X axis (column numbers). Example: '-cc=44,6940,25,4970' takes only pixels from 44 to 6940 on the vertical axis, and 25 to 4970 on the horizontal axis.")


#### 3.2. Example
```
python SpykBatch.py -d "./MyImages" -f ".tif" -r 0.5 -ct 0,30 -md 50 -spklt -efd -nh 30 -timg -tspk -cc=44,6940,25,4970
```

This line executes the `SpykBatch.py` taking `"./MyImages"` as directory; once cropped, the original images are rescaled by `0.5` (i.e., resizes the image by 0.5 in each side); the channel `0` (e.g. R in RGB, or B in BGR images) and keeps pixel values greater than `30`; and computes the Elliptical Fourier Descriptors (`-efd`) using `30` harmonics. Both the console and `logfile.txt` file in the output folder will show the time to process each image and spike within image.


#### 3.3. Output
If all the parameters are requested and satisfied, the output will be located within the given images' directory in a folder that includes the date and time of execution (YYMMDD_hhmm). Example:

```bash
SpykProps/
|-- MyImages/
|   |-- img_01.tif
|   |-- ...
|   |-- img_n.tif
|   |-- MyImages_221101_1645/
|   |   |-- SpikeSegm/
|   |   |-- SpikeletSegm/
|   |   |-- SpikeLength/
|   |   |-- Spikes_data.csv
|   |   |-- Spikelets_data.csv
|   |   |-- EFD_data.csv
|   |   |-- EucDistances_data.csv
|   |   |-- logfile.txt
|-- ...
|-- ...
```

## Instructions for single spike/image analysis
Please refer to the *Overview.ipynb* Jupyter notebook in the repository.
