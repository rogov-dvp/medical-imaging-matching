import sys

sys.path.append('../')

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

from BatchBuilder import get_batch_random, get_batch_hard
from SiameseTriplet import SiameseModel, build_siamese_triplet_network
from models.vgg import get_vgg_model
from models.ResNet import get_resnet_model
from models.InceptionNet import get_inception_model
from validation.validate_model import validate_model


def train(img_size, embeddingsize, architecture, cropped):
    path = '../data/'
    modelname = architecture + '_' + str(img_size) + '_' + str(embeddingsize)
    if cropped:
        modelname += '_cropped'
        img_rcc = np.load(path + 'img_rcc_cropped_' + str(img_size)+'.npy')
        img_lcc = np.load(path + 'img_lcc_cropped_' + str(img_size)+'.npy')
        img_rmlo = np.load(path + 'img_rmlo_cropped_' + str(img_size)+'.npy')
        img_lmlo = np.load(path + 'img_lmlo_cropped_' + str(img_size)+'.npy')
        lab_rcc = np.load(path + 'lab_rcc_cropped_' + str(img_size)+'.npy')
        lab_lcc = np.load(path + 'lab_lcc_cropped_' + str(img_size)+'.npy')
        lab_rmlo = np.load(path + 'lab_rmlo_cropped_' + str(img_size)+'.npy')
        lab_lmlo = np.load(path + 'lab_lmlo_cropped_' + str(img_size)+'.npy')
    else:
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

    train_from_scratch = True
    input_shape = (img_size,img_size,1)
    
    evaluate_every = 100 # interval for evaluating on one-shot tasks
    batch_size = 32
    selection_size = 128 # must be even
    norms_batch_size = 16
    n_iter = 5000 # No. of training iterations
    val_batch_size = 200
    
    
    if architecture == 'vgg':
        network =  get_vgg_model(img_size, embeddingsize)
    elif architecture == 'ResNet':
        network =  get_resnet_model(img_size, embeddingsize)
    elif architecture == 'Inception':
        network =  get_inception_model(img_size, embeddingsize)
    else:
        print('No valid architecture')

    siamese_network = build_siamese_triplet_network(input_shape, network)
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=Adam(learning_rate=0.00001))
    
    if not train_from_scratch:
        siamese_model.load_weights('./weights.h5')
    
    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    loss_history = []
    val_pos_history = []
    val_neg_history = []
    val_ind = []
    for i in range(n_iter):
        triplets = get_batch_hard(data_train, labels_train, batch_size, selection_size, norms_batch_size, network)
        loss = siamese_model.train_on_batch(triplets)
        loss_history.append(loss)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
            print("Train Loss: {0}".format(loss))
            
            val_triplets = get_batch_random(data_test, labels_test, val_batch_size)
            val_pos_acc, val_neg_acc = validate_model(val_triplets, network)
            val_pos_history.append(val_pos_acc)
            val_neg_history.append(val_neg_acc)
            val_ind.append(i)
            print("Avg Dist Pos: {0}".format(val_pos_acc))
            print("Avg Dist Neg: {0}".format(val_neg_acc))
            
            network.save_weights('../models/weights/Embedding_weights.{}.{}.h5'.format(i, modelname))
            siamese_model.save_weights('../models/weights/Siamese_weights.{}.{}.h5'.format(i, modelname))
    
    hist = np.zeros((n_iter, 3))
    hist[:,1:] = None
    hist[:,0] = loss_history
    hist[val_ind, 1] = val_pos_history
    hist[val_ind, 2] = val_neg_history
    
    pd.DataFrame(hist).to_csv("./loss_histories/hist.{}.csv".format(modelname))
    
    return [loss_history, val_pos_history, val_neg_acc, val_ind, (time.time()-t_start)/60.0]


# vgg_64_512_nc = train(64, 512, 'vgg', False)
# vgg_128_512_nc = train(128, 512, 'vgg', False)
# vgg_256_512_nc = train(256, 512, 'vgg', False)
# vgg_512_512_nc = train(512, 512, 'vgg', False)

# vgg_64_1024_nc = train(64, 1024, 'vgg', False)
# vgg_128_1024_nc = train(128, 1024, 'vgg', False)
# vgg_256_1024_nc = train(256, 1024, 'vgg', False)
# vgg_512_1024_nc = train(512, 1024, 'vgg', False)

# vgg_64_2048_nc = train(64, 2048, 'vgg', False)
# vgg_128_2048_nc = train(128, 2048, 'vgg', False)
# vgg_256_2048_nc = train(256, 2048, 'vgg', False)
# vgg_512_2048_nc = train(512, 2048, 'vgg', False)
          

# res_64_512_nc = train(64, 512, 'ResNet', False)
# res_128_512_nc = train(128, 512, 'ResNet', False)
# res_256_512_nc = train(256, 512, 'ResNet', False)
# res_512_512_nc = train(512, 512, 'ResNet', False)

# res_64_1024_nc = train(64, 1024, 'ResNet', False)
# res_128_1024_nc = train(128, 1024, 'ResNet', False)
# res_256_1024_nc = train(256, 1024, 'ResNet', False)
# res_512_1024_nc = train(512, 1024, 'ResNet', False)

res_64_2048_nc = train(64, 2048, 'ResNet', False)
print(res_64_2048_nc)
res_128_2048_nc = train(128, 2048, 'ResNet', False)
print(res_64_2048_nc)
print(res_128_2048_nc)
res_256_2048_nc = train(256, 2048, 'ResNet', False)
print(res_64_2048_nc)
print(res_128_2048_nc)
print(res_256_2048_nc)
res_512_2048_nc = train(512, 2048, 'ResNet', False)

print(res_64_2048_nc)
print(res_128_2048_nc)
print(res_256_2048_nc)
print(res_512_2048_nc)


# inc_64_512_nc = train(64, 512, 'Inception', False)
# inc_128_512_nc = train(128, 512, 'Inception', False)
# inc_256_512_nc = train(256, 512, 'Inception', False)
# inc_512_512_nc = train(512, 512, 'Inception', False)

# inc_64_1024_nc = train(64, 1024, 'Inception', False)
# inc_128_1024_nc = train(128, 1024, 'Inception', False)
# inc_256_1024_nc = train(256, 1024, 'Inception', False)
# inc_512_1024_nc = train(512, 1024, 'Inception', False)

# vgg_64_2048_nc = train(64, 2048, 'Inception', False)
# vgg_128_2048_nc = train(128, 2048, 'Inception', False)
# vgg_256_2048_nc = train(256, 2048, 'Inception', False)
# vgg_512_2048_nc = train(512, 2048, 'Inception', False)
