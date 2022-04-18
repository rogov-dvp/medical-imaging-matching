import os
import cv2
import numpy as np

processed_path = "../test_images_kaggle/processed_images"
unprocessed_images = "../test_images_kaggle/images"
filenames = []


def check_imgs(path):
    """
        Check if the image is processed.
        Return filepath if processed or an empty string if not.
        """
    temp_list = []

    # iterate over files in
    # that directory
    for root, dirs, files in os.walk(path):
        for file in files:
            if "jpg" in file:
                temp = os.path.join(root, file)
                temp_list.append(temp)
                filenames.append(file)
    return temp_list


def load_image(image_path):
    """
        Load image into script
        """
    return cv2.imread(image_path)


def resize_image(image):
    """
    resize the image to 256x256
    """
    # let's downscale the image using new  width and height
    down_width = 256
    down_height = 256
    down_points = (down_width, down_height)
    resized_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
    return resized_down


def save_image(image, filename):
    """
    Save image as numpy array
    MG_LCC, MG_RCC, MG_RMLO and MG_LMLO need to be saved according to mammogram type
    """
    name = filename.split(".")[0]
    if "CC" in name and "L" in name:
        # saved = cv2.imwrite(processed_path + "/lcc/" + name, image)
        np.save(processed_path + "/lcc/" + name, image)
    elif "CC" in name and "R" in name:
        # saved = cv2.imwrite(processed_path + "/rcc/" + name, image)
        np.save(processed_path + "/rcc/" + name, image)
    elif "MLO" in name and "R" in name:
        # saved = cv2.imwrite(processed_path + "/rmlo/" + name, image)
        np.save(processed_path + "/rmlo/" + name, image)
    elif "MLO" in name and "L" in name:
        # saved = cv2.imwrite(processed_path + "/lmlo/" + name, image)
        np.save(processed_path + "/lmlo/" + name, image)
    else:
        # saved = cv2.imwrite(processed_path + "/" + name, image)
        np.save(processed_path + "/" + name, image)


unprocessed_imgs = check_imgs(unprocessed_images)
for i in range(len(unprocessed_imgs)):
    img = load_image(unprocessed_imgs[i])
    resized = resize_image(img)
    save_image(resized, filenames[i])
