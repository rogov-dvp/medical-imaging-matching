import numpy as np
import time

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.utils import plot_model
import keras

from CNNTripletModel import build_network, build_model
from BatchBuilder import get_batch_random
from Evaluate import calculate_avg_dist

train_from_scratch = True

img_size = 128
input_shape = (img_size,img_size,1)

evaluate_save_every = 10
batch_size = 50
test_batch_size = 20

# Adjust
path = ''
img_rcc = np.load(path + 'img_rcc_' + str(img_size)+'.npy')
img_lcc = np.load(path + 'img_lcc_' + str(img_size)+'.npy')
img_rmlo = np.load(path + 'img_rmlo_' + str(img_size)+'.npy')
img_lmlo = np.load(path + 'img_lmlo_' + str(img_size)+'.npy')
lab_rcc = np.load(path + 'lab_rcc_' + str(img_size)+'.npy')
lab_lcc = np.load(path + 'lab_lcc_' + str(img_size)+'.npy')
lab_rmlo = np.load(path + 'lab_rmlo_' + str(img_size)+'.npy')
lab_lmlo = np.load(path + 'lab_lmlo_' + str(img_size)+'.npy')

rcc_data_train, rcc_data_test, rcc_labels_train, rcc_labels_test = train_test_split(img_rcc, lab_rcc, test_size=0.2, random_state=42)
lcc_data_train, lcc_data_test, lcc_labels_train, lcc_labels_test = train_test_split(img_lcc, lab_lcc, test_size=0.2, random_state=42)
rmlo_data_train, rmlo_data_test, rmlo_labels_train, rmlo_labels_test = train_test_split(img_rmlo, lab_rmlo, test_size=0.2, random_state=42)
lmlo_data_train, lmlo_data_test, lmlo_labels_train, lmlo_labels_test = train_test_split(img_lmlo, lab_lmlo, test_size=0.2, random_state=42)

data_train = [rcc_data_train, lcc_data_train, rmlo_data_train, lmlo_data_train]
labels_train = [rcc_labels_train, lcc_labels_train, rmlo_labels_train, lmlo_labels_train]
data_test = [rcc_data_test, lcc_data_test, rmlo_data_test, lmlo_data_test]
labels_test = [rcc_labels_test, lcc_labels_test, rmlo_labels_test, lmlo_labels_test]


if train_from_scratch:
    network = build_network(input_shape,embeddingsize=100)
else:
    network = keras.load_model('Network')
    
network_train = build_model(input_shape,network)
optimizer = Adam(lr = 0.00006)
network_train.compile(loss=None,optimizer=optimizer)
network_train.summary()
print(network_train.metrics_names)


t_start = time.time()
for i in range(3):
    #triplets = get_batch_hard(200,16,16,network)
    triplets = get_batch_random(data_train, labels_train, batch_size)
    loss = network_train.train_on_batch(triplets, None)
    print(loss)
    if i % evaluate_save_every == 0:
        network.save('Network')
        time_passed = (time.time()-t_start)/60.0
        test_dat = get_batch_random(data_test, labels_test, test_batch_size)
        test_dist_p = calculate_avg_dist(network, test_dat[0], test_dat[1])
        test_dist_n = calculate_avg_dist(network, test_dat[0], test_dat[2])
        print("\n ------------- \n")
        print("Time for {0} iterations: {1:.1f} mins, Train Loss: {2}, Avg. Distance P: {3:.4f}, Avg. Distance N: {4:.4f}".format(i, time_passed, loss, test_dist_p, test_dist_n))
    