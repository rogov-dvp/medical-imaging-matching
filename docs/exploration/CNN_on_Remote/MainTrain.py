import numpy as np
import time

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.utils import plot_model

from CNNTripletModel import build_network, build_model
from BatchBuilder import get_batch_random


input_shape = (128,128,1)

evaluate_every = 5
n_val = 5
batch_size = 200

# Adjust
path = ''
img_size = 128
img_rcc = np.load(path + 'img_rcc_' + str(img_size))
img_lcc = np.load(path + 'img_lcc_' + str(img_size))
img_rmlo = np.load(path + 'img_rmlo_' + str(img_size))
img_lmlo = np.load(path + 'img_lmlo_' + str(img_size))
lab_rcc = np.load(path + 'lab_rcc_' + str(img_size))
lab_lcc = np.load(path + 'lab_lcc_' + str(img_size))
lab_rmlo = np.load(path + 'lab_rmlo_' + str(img_size))
lab_lmlo = np.load(path + 'lab_lmlo_' + str(img_size))

rcc_data_train, rcc_data_test, rcc_labels_train, rcc_labels_test = train_test_split(img_rcc, lab_rcc, test_size=0.2, random_state=42)
lcc_data_train, lcc_data_test, lcc_labels_train, lcc_labels_test = train_test_split(img_lcc, lab_lcc, test_size=0.2, random_state=42)
rmlo_data_train, rmlo_data_test, rmlo_labels_train, rmlo_labels_test = train_test_split(img_rmlo, lab_rmlo, test_size=0.2, random_state=42)
lmlo_data_train, lmlo_data_test, lmlo_labels_train, lmlo_labels_test = train_test_split(img_lmlo, lab_lmlo, test_size=0.2, random_state=42)

data_train = [rcc_data_train, lcc_data_train, rmlo_data_train, lmlo_data_train]
labels_train = [rcc_labels_train, lcc_labels_train, rmlo_labels_train, lmlo_labels_train]


network = build_network(input_shape,embeddingsize=10)
network_train = build_model(input_shape,network)
optimizer = Adam(lr = 0.00006)
network_train.compile(loss=None,optimizer=optimizer)
network_train.summary()
plot_model(network_train,show_shapes=True, show_layer_names=True, to_file='02 model.png')
print(network_train.metrics_names)
network_train.load_weights('mnist-160k_weights.h5')


t_start = time.time()
n_iteration = 0
for i in range(30):
    #triplets = get_batch_hard(200,16,16,network)
    triplets = get_batch_random(data_train, labels_train, batch_size)
    loss = network_train.train_on_batch(triplets, None)
    print(loss)
    # n_iteration += 1
    # if i % evaluate_every == 0:
    #     print("\n ------------- \n")
    #     print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration))
    #     probs,yprob = compute_probs(network,test_images[:n_val,:,:,:],y_test_origin[:n_val])