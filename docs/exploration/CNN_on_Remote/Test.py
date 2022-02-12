import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from BatchBuilder import get_batch_random
from reshape_to_128 import reshape_to_128

img_size = 28
batch_size = 1

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

triplets = get_batch_random(data_train, labels_train, batch_size)
triplets = reshape_to_128(triplets)

img_0 = triplets[0][0,:,:,:]
img_1 = triplets[1][0,:,:,:]
img_2 = triplets[2][0,:,:,:]

plt.imshow(img_0, interpolation='nearest')
plt.show()
plt.imshow(img_1, interpolation='nearest')
plt.show()
plt.imshow(img_2, interpolation='nearest')
plt.show()




