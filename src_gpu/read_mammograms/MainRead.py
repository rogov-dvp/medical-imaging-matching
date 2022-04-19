import numpy as np
from Read_images import read_img_to_array
from matplotlib import pyplot as plt

#Adjust
root_dir = '/home/capstone/CADAnonymized'
store_dir = '../data/'

img_size = 64
crop = True
img_rcc,img_lcc,img_rmlo,img_lmlo,lab_rcc,lab_lcc,lab_rmlo,lab_lmlo = read_img_to_array(root_dir, img_size, crop) 


if crop:
    np.save(store_dir + 'img_rcc_cropped_' + str(img_size),img_rcc)
    np.save(store_dir + 'img_lcc_cropped_' + str(img_size),img_lcc)
    np.save(store_dir + 'img_rmlo_cropped_' + str(img_size),img_rmlo)
    np.save(store_dir + 'img_lmlo_cropped_' + str(img_size),img_lmlo)
    
    np.save(store_dir + 'lab_rcc_cropped_' + str(img_size),lab_rcc)
    np.save(store_dir + 'lab_lcc_cropped_' + str(img_size),lab_lcc)
    np.save(store_dir + 'lab_rmlo_cropped_' + str(img_size),lab_rmlo)
    np.save(store_dir + 'lab_lmlo_cropped_' + str(img_size),lab_lmlo)
else:
    np.save(store_dir + 'img_rcc_' + str(img_size),img_rcc)
    np.save(store_dir + 'img_lcc_' + str(img_size),img_lcc)
    np.save(store_dir + 'img_rmlo_' + str(img_size),img_rmlo)
    np.save(store_dir + 'img_lmlo_' + str(img_size),img_lmlo)
    
    np.save(store_dir + 'lab_rcc_' + str(img_size),lab_rcc)
    np.save(store_dir + 'lab_lcc_' + str(img_size),lab_lcc)
    np.save(store_dir + 'lab_rmlo_' + str(img_size),lab_rmlo)
    np.save(store_dir + 'lab_lmlo_' + str(img_size),lab_lmlo)


img_size = 128
crop = True
img_rcc,img_lcc,img_rmlo,img_lmlo,lab_rcc,lab_lcc,lab_rmlo,lab_lmlo = read_img_to_array(root_dir, img_size, crop) 


if crop:
    np.save(store_dir + 'img_rcc_cropped_' + str(img_size),img_rcc)
    np.save(store_dir + 'img_lcc_cropped_' + str(img_size),img_lcc)
    np.save(store_dir + 'img_rmlo_cropped_' + str(img_size),img_rmlo)
    np.save(store_dir + 'img_lmlo_cropped_' + str(img_size),img_lmlo)
    
    np.save(store_dir + 'lab_rcc_cropped_' + str(img_size),lab_rcc)
    np.save(store_dir + 'lab_lcc_cropped_' + str(img_size),lab_lcc)
    np.save(store_dir + 'lab_rmlo_cropped_' + str(img_size),lab_rmlo)
    np.save(store_dir + 'lab_lmlo_cropped_' + str(img_size),lab_lmlo)
    
img_size = 256
crop = True
img_rcc,img_lcc,img_rmlo,img_lmlo,lab_rcc,lab_lcc,lab_rmlo,lab_lmlo = read_img_to_array(root_dir, img_size, crop) 


if crop:
    np.save(store_dir + 'img_rcc_cropped_' + str(img_size),img_rcc)
    np.save(store_dir + 'img_lcc_cropped_' + str(img_size),img_lcc)
    np.save(store_dir + 'img_rmlo_cropped_' + str(img_size),img_rmlo)
    np.save(store_dir + 'img_lmlo_cropped_' + str(img_size),img_lmlo)
    
    np.save(store_dir + 'lab_rcc_cropped_' + str(img_size),lab_rcc)
    np.save(store_dir + 'lab_lcc_cropped_' + str(img_size),lab_lcc)
    np.save(store_dir + 'lab_rmlo_cropped_' + str(img_size),lab_rmlo)
    np.save(store_dir + 'lab_lmlo_cropped_' + str(img_size),lab_lmlo)
    
img_size = 512
crop = True
img_rcc,img_lcc,img_rmlo,img_lmlo,lab_rcc,lab_lcc,lab_rmlo,lab_lmlo = read_img_to_array(root_dir, img_size, crop) 


if crop:
    np.save(store_dir + 'img_rcc_cropped_' + str(img_size),img_rcc)
    np.save(store_dir + 'img_lcc_cropped_' + str(img_size),img_lcc)
    np.save(store_dir + 'img_rmlo_cropped_' + str(img_size),img_rmlo)
    np.save(store_dir + 'img_lmlo_cropped_' + str(img_size),img_lmlo)
    
    np.save(store_dir + 'lab_rcc_cropped_' + str(img_size),lab_rcc)
    np.save(store_dir + 'lab_lcc_cropped_' + str(img_size),lab_lcc)
    np.save(store_dir + 'lab_rmlo_cropped_' + str(img_size),lab_rmlo)
    np.save(store_dir + 'lab_lmlo_cropped_' + str(img_size),lab_lmlo)