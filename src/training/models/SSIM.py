"""
Comparing images using ORB/SIFT feature detectors
and structural similarity index.
@author: Sreenivas Bhattiprolu
https://github.com/bnsreenu/python_for_microscopists/blob/master/191_measure_img_similarity.py

Based on original author: Sreenivas Bhattiprolu
https://youtu.be/16s3Pi1InPU
https://github.com/bnsreenu/python_for_microscopists/blob/master/191_measure_img_similarity.py

For cli (not by author^):
https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python
"""

from skimage.metrics import structural_similarity
from skimage.transform import resize  # SSIM
import cv2

# import argparse      #python3 script.py --input original.png --output modified.png


# CLI for us to call these types of files:

# input
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True, help="File of the image that will be compared")
# ap.add_argument("-s", "--second", required=True, help="File of the image that will be used to compare")
# args = vars(ap.parse_args())

# PARAMETERS TO SET (INPUT) | start ---------------------------
# image paths (same dimensions preferred):
img1_path = (
    "medical-imaging-matching/test_images_kaggle/images/2016_BC003122_ MLO_L.jpg"
)
img2_path = (
    "medical-imaging-matching/test_images_kaggle/images/2016_BC014002_ MLO_L.jpg"
)

# PARAMETERS TO SET (INPUT) | end ---------------------------

# FUNCTIONS start -------------
def get_sim(img1, img2):
    ssim, diff = structural_similarity(img1, img2, full=True)
    return ssim


# FUNCTIONS end ------------

# Get images
img1 = cv2.imread(img1_path, 0)
img2 = cv2.imread(img2_path, 0)

# Resize img2 to match the W x H dimensions of img1. Best to not resize and have the same dimensions.
if img2.shape != img2.shape:
    img2 = resize(
        img2, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True
    )

# Call SSIM function
ssim = get_sim(img1, img2)
print("Similarity using SSIM is {:.3f}%\n".format(ssim * 100))
