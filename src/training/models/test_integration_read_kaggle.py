import numpy as np
from read_preprocessed import read_imgs
import unittest
import os


class TestKaggleReadIntegration(unittest.TestCase):
    def read_images_test(self):
        # testing to make sure the response is what we expect
        processed_dir = "../../../test_images_kaggle/processed_images"
        root_dir = "../../../test_images_kaggle/images"

        img_size = 28
        # img_rcc,img_lcc,img_rmlo,img_lmlo,lab_rcc,lab_lcc,lab_rmlo,lab_lmlo = read_img_to_array(root_dir, img_size)
        img_rcc, img_lcc, img_rmlo, img_lmlo, lab_rcc, lab_lcc, lab_rmlo, lab_lmlo = read_imgs(
            root_dir, processed_dir, img_size
        )
        imgs = [
            file
            for file in os.listdir(processed_dir + "/rcc")
            if os.path.isfile(os.path.join(processed_dir + "/rcc", file))
            and not file.startswith(".")
        ]
        self.assertEqual(len(imgs), len(lab_rcc))
