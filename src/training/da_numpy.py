import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os


class DataAug:
    def __init__(self, img_array):
        self.img_array = img_array

    def show_image(self, img):
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fill(self, img, h, w):
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def horizontal_shift(self, ratio=0.0):
        img = self.img_array
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

    def vertical_shift(self, ratio=0.0):
        img = self.img_array
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

    def blur(self):
        """
        Blur the image.
        """
        return cv2.GaussianBlur(self.img_array, (11, 11), 0)

    def horizontal_flip(self):
        return np.fliplr(self.img_array)

    def vertical_flip(self):
        return np.flipud(self.img_array)

    def rotation(self, angle):
        angle = int(random.uniform(-angle, angle))
        h, w = self.img_array.shape[:2]
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
        self.img_array = cv2.warpAffine(self.img_array, M, (w, h))
        return self.img_array

    def zoom(self, value):
        img = self.img_array
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

    def horizontal_shift(self, ratio=0.0):
        img = self.img_array
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

    def vertical_shift(self, ratio=0.0):
        img = self.img_array
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

    def brightness(self, low, high):
        img = self.img_array
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
