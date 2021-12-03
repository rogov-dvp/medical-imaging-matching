"""
File to preprocess image data.
"""
import os


class PreprocessData:
    def __init__(self, filename):
        self.filename = filename
        self.processed_path = "test_images_kaggle/processed_images"

    def check_processed(self):
        """
        Check if the image is processed.
        Return filepath if processed or an empty string if not.
        """
        dirs = []
        # iterate over files in
        # that directory
        for root, dirs, files in os.walk(self.processed_path):
            for filename in files:
                if filename == self.filename:
                    dirs.append(os.path.join(root, filename))
        return dirs

    def load_image(self, image_path):
        """
        Load image into script
        """

    def resize_image(self, image):
        """
        resize the image to 256x256
        """

    def save_image(self, image):
        """
        Save image as numpy array
        """

    def process_image(self):
        """
        Method to process the image
        1. Resize image
        2. Save as numpy array into folder based on mammogram type
        """
        processed_file_path = self.check_processed()
        if len(processed_file_path) != 0:
            return processed_file_path
        else:

            return ""


preprocess = PreprocessData("2016_BC003122_ CC_L")
preprocess.process_image()
