"""
Comparing images using ORB feature detectors
and structural similarity index.

Original author: Sreenivas Bhattiprolu
https://youtu.be/16s3Pi1InPU
https://github.com/bnsreenu/python_for_microscopists/blob/master/191_measure_img_similarity.py
"""

import cv2

# TODO: PARAMETERS TO SET (INPUT):
# image paths:
img1_path = (
    "/Users/alexa/Documents/UBCO/COSC499/medical-imaging-matching/test_images_kaggle/images/2016_BC003122_ MLO_L.jpg"
)
img2_path = (
    "/Users/alexa/Documents/UBCO/COSC499/medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
)

# Distance is how far the program will look for similar regions around keypoints. The higher
# the distance, the more "lenient" the program will be to matching. distance_input between 0-100. 
# Ie. basically distance of 100 will always match
distance_input = 0.5

# FUNCTIONS start -------------------------------------------
# Works well with images of different dimensions
def orb_sim(img1, img2):
    # Create ORB object with cv2
    orb = cv2.ORB_create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # perform matches.
    matches = bf.match(desc_a, desc_b)

    # "distane_input" variable tells the program how far it should look for similar regions with distance. Goes from 0 to 100.
    # Higher distance means more chances to seeing similar regions. So 100 == identical.
    similar_regions = [i for i in matches if i.distance < distance_input]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


# FUNCTIONS end -------------------------------------------

# SCRIPT start --------------------------------------------

# ORB
# Get images
img1 = cv2.imread(img1_path, 0)
img2 = cv2.imread(img2_path, 0)

# Call ORB simulator
orb_similarity = orb_sim(img1, img2)  # 1.0 means identical. Lower = not similar

# Print results
print("Similarity using ORB is {:.3f}%\n".format(orb_similarity * 100))

# SCRIPT end --------------------------------------------
