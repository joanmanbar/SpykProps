# SpykProps

### Author:
Joan Bareto Ortiz (jbarreto@umn.edu)

This repository contains the Python code to collect spike (inflorescences) properties from images of perennial ryegrass. The code was written for images collected with a flatbed scanner at 600 dpi using a black velvet as background. Differences in image size, resolution, or background can affect the output of the analysis. Please refer to the Overview.ipynb file to use SpykProps on single images using Jupyter-lab, or follow the instructions below for *batch processing*.

## Instructions for batch processing
These instructions

### 1. Clone repository
```
{
  git clone https://github.com/joanmanbar/SpykProps.git
  cd SpykProps
}
```

### 3. Add the images folder to the `SpykProps` repository
For example, if the folder is called 'MyImages', the structure for the current directory should look as follows:

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

### 2. Install requirements
```
pip install -r requirement.txt
```
*Note*: If a module (e.g. 'mymodule') is missing, install it as follows:

```
pip install mymodule
```

### 3. Execute batch function
Check the help menu
```
python SpykBatch.py -h
```
Then adjust the parameters as needed and run the function accordingly.
Example:
```
python SpykBatch.py -d ".\MyImages" -r 0.5 -ct 0,30 -efd True -nh 30
```

This line executes the `SpykBatch.py` taking `".\MyImages"` as directory; `0.5` as rescale factor (i.e., resizes the image by 0.5 in each side); the channel `0` (e.g. R in RGB, or B in BGR images) and keeps pixel values greater than `30`; and computes the Elliptical Fourier Descriptors (`-efd`) using `30` harmonics.
