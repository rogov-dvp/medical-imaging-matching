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
        

        #Pick negative image of different patient different from different patient

        while data_labels[neg_ind[i]] == data_labels[pos_ind[i]]:
            neg_ind[i] -= 1
        triplets[2][i,:,:,:] = data_train[neg_ind[i],0,:,:,:]
    return triplets

def get_batch_random(data_train, data_labels, batch_size):
    """
    Create batch of APN triplets with a complete random strategy
    """ 
    #initialize result
    h, w, c = data_train[0][0,0].shape
    
    # Inserted
    c = 3
    
    triplets=[np.zeros((batch_size, h, w, c)) for i in range(3)]
    
    cat = random.randint(0,len(data_train)-1)
    for i in range(batch_size):
        pos_ind = random.randint(0, len(data_train[cat])-1)
        neg_ind = random.randint(0, len(data_train[cat])-1)
        
        #Pick one random anchor and positive image

        triplets[0][i,:,:,:0] = data_train[cat][pos_ind,0,:,:,:0]
        triplets[1][i,:,:,:0] = data_train[cat][pos_ind,1,:,:,:0]
        triplets[0][i,:,:,:1] = data_train[cat][pos_ind,0,:,:,:0]
        triplets[1][i,:,:,:1] = data_train[cat][pos_ind,1,:,:,:0]

        
        #Pick negative image of different patient different from different patient
        while data_labels[cat][neg_ind] == data_labels[cat][pos_ind]:
            neg_ind = random.randint(0, len(data_train[cat])-1)

        triplets[2][i,:,:,0] = data_train[cat][neg_ind,0,:,:,0]
        triplets[2][i,:,:,1] = data_train[cat][neg_ind,0,:,:,0]
        triplets[2][i,:,:,2] = data_train[cat][neg_ind,0,:,:,0]

        cat += 1
        cat = cat % len(data_train)
        
    return triplets


    def compute_dist(a,b):
        return np.sum(np.square(a-b))

    def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train"):

#     """
#     Create batch of APN "hard" triplets
    """
     Arguments:
     draw_batch_size -- integer : number of initial randomly taken samples   
     hard_batchs_size -- interger : select the number of hardest samples to keep
     norm_batchs_size -- interger : number of random samples to add

     Returns:
     triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    
    if s == "train":
         X = dataset_train
    else:
         X = dataset_test

    m, w, h,c = X[0].shape
    
    
#     Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,s)
    
#     Step 2 : compute the loss with current network : d(A,P)-d(A,N). 
#     The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))
    
#     Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])
    
#     Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A - P), axis=1) - np.sum(np.square(A - N), axis=1)
    
#     Sort by distance (high distance first) and take the 
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    
#     Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection), norm_batchs_size,
    replace=False,)
    
    selection = np.append(selection,selection2)
    
    triplets = [
        studybatch[0][selection,:,:,:], 
        studybatch[1][selection,:,:,:], 
        studybatch[2][selection,:,:,:],
        ]
    
    return triplets


    def build_network(input_shape, embeddingsize):
    '''
    Define the neural network to learn image similarity
    Input : 
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our images   
    '''  

    #CNN
        network = Sequential()
#       network.add(Conv2D(128, (7,7), activation='relu',
#                       input_shape=input_shape,
#                        kernel_initializer='he_uniform',
#                        kernel_regularizer=l2(2e-4)))
#        network.add(MaxPooling2D())
#        network.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
#                        kernel_regularizer=l2(2e-4)))
#        network.add(MaxPooling2D())
#        network.add(Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',
#                        kernel_regularizer=l2(2e-4)))
#        network.add(Flatten())
#        network.add(Dense(4096, activation='relu',
#                    kernel_regularizer=l2(1e-3),
#                    kernel_initializer='he_uniform'))
    
    
#        network.add(Dense(embeddingsize, activation=None,
#                    kernel_regularizer=l2(1e-3),
#                    kernel_initializer='he_uniform'))
    
    #Force the encoding to live on the d-dimentional hypershpere
#    network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))

    return network

    # Loss function L=max(d(A,P)−d(A,N)+margin,0)
    class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def build_model(input_shape, network, margin=0.2):
    '''
    Define the Keras Model for training 
        Input : 
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    
    '''
     # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    
    # return the model
    return network_train

   
  
    



  