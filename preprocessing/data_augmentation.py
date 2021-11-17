import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os


class DataAug:
    def __init__(self, file_dir):
        self.imgs_dir = file_dir

    def get_files(self):
        """
            This function gets all the pure file names
            """
        files = []
        for file in os.listdir(self.imgs_dir):
            if file.endswith(".jpg"):
                files.append(file)

        return files

    def read_image(self, img_name):
        """
        Read Image into Program
        """
        return cv2.imread(self.imgs_dir + "/" + img_name)

    def show_image(self, img):
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fill(self, img, h, w):
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def horizontal_shift(self, img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print("Value should be less than 1 and greater than 0")
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = w * ratio
        if ratio > 0:
            img = img[:, : int(w - to_shift), :]
        if ratio < 0:
            img = img[:, int(-1 * to_shift) :, :]
        img = self.fill(img, h, w)
        return img

    def vertical_shift(self, img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print("Value should be less than 1 and greater than 0")
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = h * ratio
        if ratio > 0:
            img = img[: int(h - to_shift), :, :]
        if ratio < 0:
            img = img[int(-1 * to_shift) :, :, :]
        img = self.fill(img, h, w)
        return img

    def brightness(self, img, low, high):
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def zoom(self, img, value):
        if value > 1 or value < 0:
            print("Value for zoom should be less than 1 and greater than 0")
            return img
        value = random.uniform(value, 1)
        h, w = img.shape[:2]
        h_taken = int(value * h)
        w_taken = int(value * w)
        h_start = random.randint(0, h - h_taken)
        w_start = random.randint(0, w - w_taken)
        img = img[h_start : h_start + h_taken, w_start : w_start + w_taken, :]
        img = self.fill(img, h, w)
        return img

    def channel_shift(self, img, value):
        value = int(random.uniform(-value, value))
        img = img + value
        img[:, :, :][img[:, :, :] > 255] = 255
        img[:, :, :][img[:, :, :] < 0] = 0
        img = img.astype(np.uint8)
        return img

    def horizontal_flip(self, img, flag):
        if flag:
            return cv2.flip(img, 1)
        else:
            return img

    def vertical_flip(self, img, flag):
        if flag:
            return cv2.flip(img, 0)
        else:
            return img

    def rotation(self, img, angle):
        angle = int(random.uniform(-angle, angle))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        return img


da = DataAug("../test_images_kaggle/images")

# get all image names
image_files = da.get_files()

# data augment each image in the images folder
for img_name in image_files:
    # Read in image
    image = da.read_image(img_name)

    # horizontal shift
    hs_image = da.horizontal_shift(image, 0.7)
    saved = cv2.imwrite(
        "../test_images_kaggle/preprocessed_images/hrshift_" + img_name, hs_image
    )
    # da.show_image(hs_image)

    # vertical shift
    img = da.vertical_shift(image, 0.7)
    saved = cv2.imwrite(
        "../test_images_kaggle/preprocessed_images/vshift_" + img_name, img
    )

    # brightness
    bright_img = da.brightness(image, 0.5, 3)
    saved = cv2.imwrite(
        "../test_images_kaggle/preprocessed_images/bright_" + img_name, bright_img
    )

    # zoomed
    zoomed_img = da.zoom(image, 0.5)
    saved = cv2.imwrite(
        "../test_images_kaggle/preprocessed_images/zoomed_" + img_name, zoomed_img
    )

    # channel shift
    cshift_img = da.channel_shift(image, 60)
    saved = cv2.imwrite(
        "../test_images_kaggle/preprocessed_images/cshift_" + img_name, cshift_img
    )

    # h flip
    hflip_img = da.horizontal_flip(image, True)
    saved = cv2.imwrite(
        "../test_images/kaggle/preprocesses_images/hflip_" + img_name, hflip_img
    )

    # v flip
    vflip_img = da.vertical_flip(image, True)
    saved = cv2.imwrite(
        "../test_images_kaggle/preprocessed_images/vflip_" + img_name, vflip_img
    )

    # rotation
    rotated_img = da.rotation(image, 30)
    saved = cv2.imwrite(
        "../test_images_kaggle/preprocessed_images/rotated_" + img_name, rotated_img
    )
