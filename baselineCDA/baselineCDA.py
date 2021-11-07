# Baseline algorithm for general change detection.
# Christofer Schwartz, 2021
# Python port: David Mallasen
import cv2
import numpy as np
import tifffile as tiff

# Select the band and threshold value
band = 2
tr = 300

# Load images
IrefRaw = tiff.imread("../test_images/area2_20200610_100759_89_1061_3B_AnalyticMS_SR.tif")
IsurRaw = tiff.imread("../test_images/area2_20200620_100924_87_1066_3B_AnalyticMS_SR.tif")
IrefWork = IrefRaw.astype(np.float64)
IsurWork = IsurRaw.astype(np.float64)

# Select image band
Iref = IrefWork[:, :, band]
Isur = IsurWork[:, :, band]

# Pre-filtering
# In https://doi.org/10.1117/12.663594, data is averaged with a low-pass 
# filter to reduce influence from noise in VHF UWB SAR images. 
# The averaging also has the benefit to improve the validity of the data
# being Gaussian distributed.
IsurH = cv2.blur(Isur, (5, 5))
IrefH = cv2.blur(Iref, (5, 5))

# Calculate image difference
Idif = IsurH - IrefH

# Abs value (diferences negatives and positives) and thresholding
_, Itr = cv2.threshold(Idif, tr, 1, cv2.THRESH_BINARY)

# Morphological operations
se = np.array([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]], np.uint8)
If = cv2.erode(Itr, se)
If = cv2.dilate(If, se)

# Plot results
If = cv2.normalize(If, dst=None, alpha=0, beta=np.iinfo(np.uint16).max, norm_type=cv2.NORM_MINMAX)
cv2.imwrite("cda_out.png", If)