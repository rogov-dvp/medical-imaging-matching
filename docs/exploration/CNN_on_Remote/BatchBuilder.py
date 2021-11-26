import numpy as np
import random

# Build Triplets
def get_batch_random_demo(data_train, data_labels, batch_size):
    """
    Create batch of APN triplets with a complete random strategy
    
    Arguments:
    batch_size -- integer 

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    no_train = data_train.shape[0] 
    h, w, c = data_train[0,0].shape
    triplets=[np.zeros((batch_size, h, w, c)) for i in range(3)]
    
    pos_ind = np.random.randint(0, no_train, size=batch_size)
    neg_ind = np.random.randint(0, no_train, size=batch_size)
    
    for i in range(batch_size):
        #Pick one random anchor and positive image
        triplets[0][i,:,:,:] = data_train[pos_ind[i],0,:,:,:]
        triplets[1][i,:,:,:] = data_train[pos_ind[i],1,:,:,:]
        
        # WE HAVE TO CHECK HERE THAT WE TAKE SAME TYPE???
        
        #Pick negative image of different patient different from different patient
        while data_labels[neg_ind[i]] == data_labels[pos_ind[i]]:
            neg_ind[i] -= 1
        triplets[2][i,:,:,:] = data_train[neg_ind[i],0,:,:,:]
    return triplets

def get_batch_random(data_train, data_labels, batch_size):
    """
    Create batch of APN triplets with a complete random strategy
    """ 
    h, w, c = data_train[0][0,0].shape
    triplets=[np.zeros((batch_size, h, w, c)) for i in range(3)]
    
    cat = random.randint(0,len(data_train)-1)
    for i in range(batch_size):
        pos_ind = random.randint(0, len(data_train[cat])-1)
        neg_ind = random.randint(0, len(data_train[cat])-1)
        
        #Pick one random anchor and positive image
        triplets[0][i,:,:,:] = data_train[cat][pos_ind,0,:,:,:]
        triplets[1][i,:,:,:] = data_train[cat][pos_ind,1,:,:,:]
        
        #Pick negative image of different patient different from different patient
        while data_labels[cat][neg_ind] == data_labels[cat][pos_ind]:
            neg_ind = random.randint(0, len(data_train[cat])-1)
        triplets[2][i,:,:,:] = data_train[cat][neg_ind,0,:,:,:]
        cat += 1
        cat = cat % len(data_train)
    return triplets

# def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train"):
#     """
#     Create batch of APN "hard" triplets
    
#     Arguments:
#     draw_batch_size -- integer : number of initial randomly taken samples   
#     hard_batchs_size -- interger : select the number of hardest samples to keep
#     norm_batchs_size -- interger : number of random samples to add

#     Returns:
#     triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
#     """
#     if s == 'train':
#         X = dataset_train
#     else:
#         X = dataset_test

#     m, w, h,c = X[0].shape
    
    
#     #Step 1 : pick a random batch to study
#     studybatch = get_batch_random(draw_batch_size,s)
    
#     #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
#     studybatchloss = np.zeros((draw_batch_size))
    
#     #Compute embeddings for anchors, positive and negatives
#     A = network.predict(studybatch[0])
#     P = network.predict(studybatch[1])
#     N = network.predict(studybatch[2])
    
#     #Compute d(A,P)-d(A,N)
#     studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)
    
#     #Sort by distance (high distance first) and take the 
#     selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    
#     #Draw other random samples from the batch
#     selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)
    
#     selection = np.append(selection,selection2)
    
#     triplets = [studybatch[0][selection,:,:,:], studybatch[1][selection,:,:,:], studybatch[2][selection,:,:,:]]
    
#     return triplets