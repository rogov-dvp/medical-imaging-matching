"""
File to preprocess image data.
"""
import os
import cv2


class PreprocessData:
    def __init__(self, filename):
        self.filename = filename
        self.processed_path = "test_images_kaggle/processed_images"
        self.unprocessed_images = "test_images_kaggle/images"

    def check_imgs(self, path):
        """
        Check if the image is processed.
        Return filepath if processed or an empty string if not.
        """
        dirs = []
        # iterate over files in
        # that directory
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename == self.filename:
                    dirs.append(os.path.join(root, filename))
        return dirs

    def load_image(self, image_path):
        """
        Load image into script
        """
        return cv2.imread(image_path)

    def resize_image(self, image):
        """
        resize the image to 256x256
        """
        # let's downscale the image using new  width and height
        down_width = 256
        down_height = 256
        down_points = (down_width, down_height)
        resized_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
        return resized_down

    def save_image(self, image):
        """
        Save image as numpy array
        MG_LCC, MG_RCC, MG_RMLO and MG_LMLO need to be saved according to mammogram type
        """
        if "MG_LCC" in self.filename:
            saved = cv2.imwrite(self.processed_path + "/lcc/" + self.filename, image)
        elif "MG_RCC" in self.filename:
            saved = cv2.imwrite(self.processed_path + "/rcc/" + self.filename, image)
        elif "MG_RMLO" in self.filename:
            saved = cv2.imwrite(self.processed_path + "/rmlo/" + self.filename, image)
        elif "MG_LMLO" in self.filename:
            saved = cv2.imwrite(self.processed_path + "/lmlo/" + self.filename, image)
        else:
            saved = cv2.imwrite(self.processed_path + "/" + self.filename, image)

    def process_image(self):
        """
        Method to process the image
        1. Resize image
        2. Save as numpy array into folder based on mammogram type
        """
        processed_file_path = self.check_imgs(self.processed_path)
        if len(processed_file_path) != 0:
            return processed_file_path
        else:
            unprocessed_imgs = self.check_imgs(self.unprocessed_images)
            img = self.load_image(unprocessed_imgs[0])
            resized = self.resize_image(img)
            self.save_image(resized)
            return "Image saved here: " + str(self.processed_path)


preprocess = PreprocessData("2016_BC003122_ CC_L.jpg")
preprocess.process_image()
