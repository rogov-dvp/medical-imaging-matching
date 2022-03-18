# regression test CLI commands. The first is to reset and set the correct values.
# The second CLI is to test if any changes occur (reset & test):
# pytest --regtest-reset <filename> 
# pytest <filename>

import cv2
from ORB import orb_sim, img1, img2

def test_general(regtest):
    value = orb_sim(img1, img2)
    print(value,file=regtest)   #Write the regression test into file.

#Testing orb_sim with dummy data. Images are identical and should return 1
def test_dummy_data_identical(regtest):
    img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = orb_sim(img1_inner,img2_inner)
    print(value,file=regtest)

#Testing orb_sim with dummy data. Both images are similar and show return a number between 0 and 1
def test_dummy_data_similar(regtest):
    img1_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC019521_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = orb_sim(img1_inner,img2_inner)
    print(value,file=regtest)   

#testing orb_sim with dummy data. Both images are opposite and should return 0
def test_dummy_data_different(regtest):
    img1_path = "medical-imaging-matching/docs/exploration/Bond3.jpg"
    img2_path = "medical-imaging-matching/test_images_kaggle/images/2017_BC011081_ MLO_L.jpg"
    #get path
    img1_inner = cv2.imread(img1_path,0)
    img2_inner = cv2.imread(img2_path,0)
    value = orb_sim(img1_inner,img2_inner) 
    print(value,file=regtest)

# Check if img1 exists (file path is correct)
def test_img1_not_none(regtest):
    print(type(img1) != None,file=regtest)

# Check if img2 exists (file path is correct)
def test_img2_not_none(regtest):
    print(type(img2) != None,file=regtest)
