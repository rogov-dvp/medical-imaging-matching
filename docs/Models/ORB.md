#  Oriented FAST and rotated BRIEF (ORB)
Created: November 19th, 2021
Last Updated: November 19th, 2021

## Description
ORB is a image feature detector which finds FAST keypoints and BRIEF descriptors of an image. These are used to compare and determine the similarity of two images. To understand this, we must understand the two
 - FAST (Features from Accelerated and Segments Test) keypoints are pixels which are neighboured with pixels of different brigthness (higher or lower). If a pixel has atleast 8 neighbouring pixels with different brightness, then it is considered a keypoint.
 - BRIEF (Binary robust independent elementary feature) descriptors are a set of binary feature vectors based on all keypoints collected. Each descriptor of each keypoint takes two random pixels around the keypoint pixel and compares the brightness. If the first pixel is brighter than the second pixel, it is assigned a bit of 1 else 0. It repeats this for 128 times to create a bit string of 128 (other variations can have more). The surrounding area which these pixels are selected from is called a patch. The patch size is dictated based on the provided "distance" parameters. Therefore, a higher distance will provide a less accurate result since the chances of two pixels being different decreases. 

Additionally, ORB adds it's own features too. During the calculation of FAST keypoints, ORB will check them at lower resolutions of the image. Keypoints that are also found in lower resolutions highlights pixel/region of the image. This makes ORB partial resolution independent. Once a keypoint is found, ORB assigns it a direction based on the surronding pixel brightness, computes descriptors (feature vector/numerical fingerprint). Thus, the keypoints have some rotation invariance.

The similarity matching occurs when the vector descriptors between images are compared.

### Comparison with CNN
ORB is a fast model but not accurate.  It does not "learn" and the associated weight (which is manually given) can really change the outcome. A positive is that CNN does not require lots of data. This is why CNN would be preferred. 

### Imports 
Possible libraries to consider in the process:
 1. openCV
 



### Implementation 
Input two images (preferably the same size) and compare them using the imported method.

### References 
- https://github.com/bnsreenu/python_for_microscopists/blob/master/191_measure_img_similarity.py
- https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf

