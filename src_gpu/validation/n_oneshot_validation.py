import sys
sys.path.append('../')

import numpy as np
import random

from models.vgg import get_vgg_model
from models.ResNet import get_resnet_model
# from models.InceptionNet import get_inception_model

def make_oneshot_task(data, data_labels, N):
    h, w, c = data[0][0,0].shape
    
    pairs=[np.zeros((N, h, w, c)) for i in range(2)]
    
    cat = random.randint(0,len(data)-1)
    targets = np.zeros(N)
    rand_pos_ind = random.randint(0,N-1)
    targets[rand_pos_ind] = 1
    pos_ind = random.randint(0, len(data[cat])-1)
    
    for i in range(N):
        neg_ind = random.randint(0, len(data[cat])-1)
        
        #Pick one random anchor and positive image
        if i==rand_pos_ind:
            pairs[0][i,:,:,0] = data[cat][pos_ind,0,:,:,0]
            pairs[1][i,:,:,0] = data[cat][pos_ind,1,:,:,0]
        else:
            #Pick negative image of different patient different from different patient
            while data_labels[cat][neg_ind] == data_labels[cat][pos_ind]:
                neg_ind = random.randint(0, len(data[cat])-1)
            rand_bin = random.randint(0, 1)
            pairs[0][i,:,:,0] = data[cat][pos_ind,0,:,:,0]
            pairs[1][i,:,:,0] = data[cat][neg_ind,rand_bin,:,:,0]
            
    return pairs, targets


def test_oneshot(data, labels, model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(data, labels, N)

        embed_a = model.predict(inputs[0])
        embed_o = model.predict(inputs[1])
        distances = np.sqrt(np.sum(np.square(embed_a - embed_o),axis=1)) / embed_a.shape[0]

        i_pred = np.argmin(distances)
        i_exp = np.argmax(targets)
        if i_pred == i_exp:
            n_correct += 1
        
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct


for i in range(3):
    for j in range(4):
        try:
            
            architecture = 'ResNet'
            img_size = 64 * 2**j
            embeddingsize = 512 * 2**i
            modelname = architecture + '_' + str(img_size) + '_' + str(embeddingsize)
            N = 20
            k = 1000
            
            # load data
            path = '../data/'
            img_rcc = np.load(path + 'img_rcc_' + str(img_size)+'.npy')
            img_lcc = np.load(path + 'img_lcc_' + str(img_size)+'.npy')
            img_rmlo = np.load(path + 'img_rmlo_' + str(img_size)+'.npy')
            img_lmlo = np.load(path + 'img_lmlo_' + str(img_size)+'.npy')
            lab_rcc = np.load(path + 'lab_rcc_' + str(img_size)+'.npy')
            lab_lcc = np.load(path + 'lab_lcc_' + str(img_size)+'.npy')
            lab_rmlo = np.load(path + 'lab_rmlo_' + str(img_size)+'.npy')
            lab_lmlo = np.load(path + 'lab_lmlo_' + str(img_size)+'.npy')
            
            data = [img_rcc, img_lcc, img_rmlo, img_lmlo]
            labels = [lab_rcc, lab_lcc, lab_rmlo, lab_lmlo]
            
            
            # get model
            # model = get_vgg_model(img_size, embeddingsize)
            model = get_resnet_model(img_size, embeddingsize)
            model.load_weights('../models/weights/Embedding_weights.{}.{}.h5'.format(4900, modelname))
            
            
            # call test_oneshot adjust so the difference for each pair is calculated
            print(img_size)
            print(embeddingsize)
            print(test_oneshot(data, labels, model, N, k))
        except:
            "Weights not found"




