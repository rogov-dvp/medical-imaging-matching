import numpy as np
from Read_images import read_img_to_array
from read_preprocessed import read_imgs

# Adjust
# root_dir = '/home/capstone/Desktop/CADAnonymized'
# root_dir = './Data'
processed_dir = "../../../test_images_kaggle/processed_images"
root_dir = "../../../test_images_kaggle/images"
store_dir = ""

img_size = 28
# img_rcc,img_lcc,img_rmlo,img_lmlo,lab_rcc,lab_lcc,lab_rmlo,lab_lmlo = read_img_to_array(root_dir, img_size)
img_rcc, img_lcc, img_rmlo, img_lmlo, lab_rcc, lab_lcc, lab_rmlo, lab_lmlo = read_imgs(
    root_dir, processed_dir, img_size
)

# np.save(store_dir + 'img_rcc_' + str(img_size),img_rcc)
# np.save(store_dir + 'img_lcc_' + str(img_size),img_lcc)
# np.save(store_dir + 'img_rmlo_' + str(img_size),img_rmlo)
# np.save(store_dir + 'img_lmlo_' + str(img_size),img_lmlo)

# np.save(store_dir + 'lab_rcc_' + str(img_size),lab_rcc)
# np.save(store_dir + 'lab_lcc_' + str(img_size),lab_lcc)
# np.save(store_dir + 'lab_rmlo_' + str(img_size),lab_rmlo)
# np.save(store_dir + 'lab_lmlo_' + str(img_size),lab_lmlo)
