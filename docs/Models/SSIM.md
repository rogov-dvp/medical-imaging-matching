#  Structural Similarity Index Measure (SSIM)
Created: November 19th, 2021
Last Updated: November 19th, 2021

## Description
SSIM is a method that quantifies the similarity of two images. It abstracts three key features to compare the images:
- Luminance
- Contrast
- Structure
The two images must have the same dimensions as each pixel of one images is measured against the other image of the same coordinates. SSIM will go through the images regionally checking rather than doing the entire image all at once for better results.

### Imports 
Possible libraries to consider in the process:
 1. openCV
 2. skimage

### Comparison with CNN
SSIM is a fast model but not accurate.  SSIM compares the images based on light and contrast of two images. A positive is that SSIm only requires two images which is possibly the maximum images we may have. However, the accuracy is more important and is why CNN would be preferred. 

### Implementation 
Input two images (preferably the same size) and compare them using the imported method.

### References 
- https://github.com/bnsreenu/python_for_microscopists/blob/master/191_measure_img_similarity.py

