import os
import numpy as np
import sys

# import pydicom
from PIL import Image


from skimage.transform import resize

sys.path.append("../../../preprocessing/")
from preprocess_data import PreprocessData


def create_array(path):
    # img = pydicom.dcmread(path)
    img = Image.open(path)
    img_arr = np.asarray(img).reshape(28, 28, 1)
    return img_arr


def read_imgs(root_dir, processed_dir, size):
    images_rcc = []
    images_lcc = []
    images_rmlo = []
    images_lmlo = []
    labels_rcc = []
    labels_lcc = []
    labels_rmlo = []
    labels_lmlo = []

    imgs = [
        file
        for file in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, file)) and not file.startswith(".")
    ]
    processed_types = [
        directory
        for directory in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, directory))
    ]

    images = []

    for id, i in enumerate(processed_types):
        rcc = []
        lcc = []
        rmlo = []
        lmlo = []
        ses_path = processed_dir + "/" + i
        images.extend(
            [
                file
                for file in os.listdir(ses_path)
                if os.path.isfile(os.path.join(ses_path, file))
                and not file.startswith(".")
            ]
        )

    image_names = [name.split(".")[0] for name in images]

    for i in imgs:
        if i.split(".")[0] not in image_names:
            print(i)
            ppd = PreprocessData(i, processed_dir, root_dir)
            images.append(ppd.process_image())

    for image in images:
        try:
            # img_in = create_array(os.path.join(ses_path, image)) / 255
            # img_in = resize(img_in, (size, size))
            # img_in = np.expand_dims(img_in, axis=2)
            if "CC_R" in image.upper():
                rcc.append(image)
            elif "CC_L" in image.upper():
                lcc.append(image)
            elif "MLO_R" in image.upper():
                rmlo.append(image)
            elif "MLO_L" in image.upper():
                lmlo.append(image)
        except:
            print(os.path.join(ses_path, image))

    for i in range(len(rcc)):
        labels_rcc.append(rcc[i].split(".")[0])
        image_path = processed_dir + "/rcc/" + rcc[i]
        images_rcc.append(np.load(image_path))

    for i in range(len(lcc)):
        labels_lcc.append(lcc[i].split(".")[0])
        image_path = processed_dir + "/lcc/" + lcc[i]
        images_lcc.append(np.load(image_path))

    for i in range(len(rmlo)):
        labels_rmlo.append(rmlo[i].split(".")[0])
        image_path = processed_dir + "/rmlo/" + rmlo[i]
        images_rmlo.append(np.load(image_path))

    for i in range(len(lmlo)):
        labels_lmlo.append(lmlo[i].split(".")[0])
        image_path = processed_dir + "/lmlo/" + lmlo[i]
        images_rmlo.append(np.load(image_path))

    return (
        images_rcc,
        images_lcc,
        images_rmlo,
        images_lmlo,
        labels_rcc,
        labels_lcc,
        labels_rmlo,
        labels_lmlo,
    )
