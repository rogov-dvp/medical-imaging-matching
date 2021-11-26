import numpy as np
import time

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.utils import plot_model

from CNNTripletModel import build_network, build_model
from BatchBuilder import get_batch_random_demo


input_shape = (28,28,1)

evaluate_every = 5
n_val = 5
batch_size = 20

data = np.load('/Users/niklastecklenburg/Desktop/Test/Data/images.npy')
labels = np.load('/Users/niklastecklenburg/Desktop/Test/Data/labels.npy')

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

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
    triplets = get_batch_random_demo(data_train, labels_train, batch_size)
    loss = network_train.train_on_batch(triplets, None)
    print(loss)
    # n_iteration += 1
    # if i % evaluate_every == 0:
    #     print("\n ------------- \n")
    #     print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration))
    #     probs,yprob = compute_probs(network,test_images[:n_val,:,:,:],y_test_origin[:n_val])