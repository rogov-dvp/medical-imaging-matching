import unittest

from data_augmentation import (
    get_files,
    read_image,
    show_image,
    fill,
    horizontal_shift,
    vertical_shift,
    brightness,
    zoom,
    channel_shift,
    horizontal_flip,
    vertical_flip,
    rotation,
)


class TestDA(unittest.TestCase):
    def test_get_files(self):
        """
        Tests if the file names are gotten
        """
        self.assertEqual()

    def test_read_image(self):
        """
        Tests if the image files are read
        """
        self.assertEqual()

    # def test_show_image(self):
    #     """
    #     Tests if the images are shown
    #     """
    #     self.assertEqual()

    def test_horizontal_shift(self):
        """
        Tests if the images are shifted horizontally
        """
        self.assertEqual()

    def test_vertical_shift(self):
        """
        Tests if the images are shifted vertically
        """
        self.assertEqual()

    def test_brightness(self):
        """
        Tests if the images are brightened
        """
        self.assertEqual()

    def test_zoom(self):
        """
        Tests if the images are zoomed
        """
        self.assertEqual()

    def test_channel_shift(self):
        """
        Tests if the channel is shifted
        """
        self.assertEqual()

    def test_horizontal_flip(self):
        """
        Tests if the image is flipped horizontally
        """
        self.assertEqual()

    def test_vertical_flip(self):
        """
        Tests if the image is flipped vertically
        """
        self.assertEqual()

    def test_rotation(self):
        """
        Tests if the image is rotated
        """
        self.assertEqual()


if __name__ == "__main__":
    unittest.main()
