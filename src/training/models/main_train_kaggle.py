import numpy as np
import time
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

# from keras.utils import plot_model
from tensorflow.keras import models

from CNNTripletModel import build_network, build_model, build_VGG16
from BatchBuilder import get_batch_random
from Evaluate import calculate_avg_dist

from SiameseTriplet import SiameseModel, build_siamese_triplet_network, build_VGG16

from reshape_to_128 import reshape_to_128

train_from_scratch = False

img_size = 28
input_shape = (img_size, img_size, 1)

evaluate_save_every = 5
batch_size = 50
test_batch_size = 20

# from read_preprocessed import read_imgs

# # Adjust
# # root_dir = '/home/capstone/Desktop/CADAnonymized'
# # root_dir = './Data'
# processed_dir = "../../../test_images_kaggle/processed_images"
# root_dir = "../../../test_images_kaggle/images"
# store_dir = ""

# img_size = 28
# # img_rcc,img_lcc,img_rmlo,img_lmlo,lab_rcc,lab_lcc,lab_rmlo,lab_lmlo = read_img_to_array(root_dir, img_size)
# img_rcc, img_lcc, img_rmlo, img_lmlo, lab_rcc, lab_lcc, lab_rmlo, lab_lmlo = read_imgs(
#     root_dir, processed_dir, img_size
# )

# Adjust
path = ""
img_rcc = np.load(path + "img_rcc_" + str(img_size) + ".npy")
img_lcc = np.load(path + "img_lcc_" + str(img_size) + ".npy")
img_rmlo = np.load(path + "img_rmlo_" + str(img_size) + ".npy")
img_lmlo = np.load(path + "img_lmlo_" + str(img_size) + ".npy")
lab_rcc = np.load(path + "lab_rcc_" + str(img_size) + ".npy")
lab_lcc = np.load(path + "lab_lcc_" + str(img_size) + ".npy")
lab_rmlo = np.load(path + "lab_rmlo_" + str(img_size) + ".npy")
lab_lmlo = np.load(path + "lab_lmlo_" + str(img_size) + ".npy")

rcc_data_train, rcc_data_test, rcc_labels_train, rcc_labels_test = train_test_split(
    img_rcc, lab_rcc, test_size=0.2, random_state=42
)
lcc_data_train, lcc_data_test, lcc_labels_train, lcc_labels_test = train_test_split(
    img_lcc, lab_lcc, test_size=0.2, random_state=42
)
rmlo_data_train, rmlo_data_test, rmlo_labels_train, rmlo_labels_test = train_test_split(
    img_rmlo, lab_rmlo, test_size=0.2, random_state=42
)
lmlo_data_train, lmlo_data_test, lmlo_labels_train, lmlo_labels_test = train_test_split(
    img_lmlo, lab_lmlo, test_size=0.2, random_state=42
)

data_train = [rcc_data_train, lcc_data_train, rmlo_data_train, lmlo_data_train]
labels_train = [
    rcc_labels_train,
    lcc_labels_train,
    rmlo_labels_train,
    lmlo_labels_train,
]
data_test = [rcc_data_test, lcc_data_test, rmlo_data_test, lmlo_data_test]
labels_test = [rcc_labels_test, lcc_labels_test, rmlo_labels_test, lmlo_labels_test]


img_size = 128
input_shape = (img_size, img_size, 3)
embeddingsize = 100

if train_from_scratch:
    # network = build_network(input_shape,embeddingsize=100)
    network = build_VGG16(input_shape=input_shape, embeddingsize=100)
else:
    network = models.load_model("Network")


# def dummy_loss(y_true, y_pred):
#     return y_pred

# network_train = build_model(input_shape,network)
# optimizer = Adam(lr = 0.00006)
# network_train.compile(loss=dummy_loss, optimizer=optimizer)
# network_train.summary()
# print(network_train.metrics_names)

siamese_network = build_siamese_triplet_network(input_shape, embeddingsize, network)
siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=Adam(0.0001))

t_start = time.time()
for i in range(100):
    # triplets = get_batch_hard(200,16,16,network)
    triplets = get_batch_random(data_train, labels_train, batch_size)

    # inserted for local run (resize the images to 128x128)
    triplets = reshape_to_128(triplets)

    loss = siamese_model.train_on_batch(triplets, None)
    print(loss)
    if i % evaluate_save_every == 0:
        network.save("Network")
        time_passed = (time.time() - t_start) / 60.0
        test_dat = get_batch_random(data_test, labels_test, test_batch_size)
        test_dat = reshape_to_128(test_dat)
        test_dist_p = calculate_avg_dist(network, test_dat[0], test_dat[1])
        test_dist_n = calculate_avg_dist(network, test_dat[0], test_dat[2])

        img_0 = test_dat[0][0, :, :, :]
        img_1 = test_dat[1][0, :, :, :]
        img_2 = test_dat[2][0, :, :, :]

        print("\n ------------- \n")
        print(
            "Time for {0} iterations: {1:.1f} mins, Train Loss: {2}, Avg. Distance P: {3:.4f}, Avg. Distance N: {4:.4f}".format(
                i, time_passed, loss, test_dist_p, test_dist_n
            )
        )
