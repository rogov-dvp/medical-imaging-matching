import unittest
import os

from data_augmentation import DataAug


class TestDA(unittest.TestCase):
    def test_get_files(self):
        """
        Tests if the file names are gotten
        """
        da = DataAug("../test_images_kaggle/images")
        files = da.get_files()
        self.assertEqual(len(files), 50)

    def test_read_image(self):
        """
        Tests if the image files are read
        """
        da = DataAug("../test_images_kaggle/images")
        files = da.get_files()
        img = da.read_image(files[0])
        self.assertNotEqual(img.size, 0)

    # def test_show_image(self):
    #     """
    #     Tests if the images are shown
    #     """
    #     self.assertEqual()

    # def test_horizontal_shift(self):
    #     """
    #     Tests if the images are shifted horizontally
    #     """
    #     self.assertEqual()

    # def test_vertical_shift(self):
    #     """
    #     Tests if the images are shifted vertically
    #     """
    #     self.assertEqual()

    # def test_brightness(self):
    #     """
    #     Tests if the images are brightened
    #     """
    #     self.assertEqual()

    # def test_zoom(self):
    #     """
    #     Tests if the images are zoomed
    #     """
    #     self.assertEqual()

    # def test_channel_shift(self):
    #     """
    #     Tests if the channel is shifted
    #     """
    #     self.assertEqual()

    # def test_horizontal_flip(self):
    #     """
    #     Tests if the image is flipped horizontally
    #     """
    #     self.assertEqual()

    # def test_vertical_flip(self):
    #     """
    #     Tests if the image is flipped vertically
    #     """
    #     self.assertEqual()

    # def test_rotation(self):
    #     """
    #     Tests if the image is rotated
    #     """
    #     self.assertEqual()


if __name__ == "__main__":
    unittest.main()
