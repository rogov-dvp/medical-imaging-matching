# calculate the similarity for two images (as np arrays)
import sys
sys.path.append('../')
import numpy as np
import cv2

from models.vgg import get_vgg_model

def calculate_similarity(img_1, img_2):
    # calculate the similarity for two images (images given as np arrays)
    
    img_size = 512
    embeddingsize = 1024
    modelname = 'vgg_512_1024'
    
    # adjust to grayscale 
    if img_1.shape[2] == 3:
        img_1 = cv2.cv2Color(img_1, cv2.COLOR_BGR2GRAY)
    if img_2.shape[2] == 3:
        img_2 = cv2.cv2Color(img_2, cv2.COLOR_BGR2GRAY)
    
    # adjust img size to match (512,512)
    img_1 = cv2.resize(img_1, (512,512))
    img_2 = cv2.resize(img_2, (512,512))
    
    im1 = np.zeros((img_size, img_size, 1))
    im2 = np.zeros((img_size, img_size, 1))
    im1[:,:,0] = img_1
    im2[:,:,0] = img_2
    
    img_1 = np.expand_dims(im1, 0)
    img_2 = np.expand_dims(im2, 0)
    

    avg_dist_pos = 1.3
    avg_dist_neg = 2.5
    
    model = model = get_vgg_model(img_size, embeddingsize)
    model.load_weights('../models/weights/Embedding_weights.{}.{}.h5'.format(4900, modelname))
    
    embed_1 = model.predict(img_1)
    embed_2 = model.predict(img_2)
    
    dist = sum(np.sqrt(np.sum(np.square(embed_1 - embed_2),axis=1))) / embed_1.shape[0]
    
    if dist < avg_dist_pos:
        dist = avg_dist_pos
    if dist > avg_dist_neg:
        dist = avg_dist_neg
        
    dist = dist - avg_dist_pos
    avg_dist_neg = avg_dist_neg - avg_dist_pos

    dist = dist / avg_dist_neg   
    dist = 1 - dist
    perc = round(100 * dist, 1)
    
    return perc
        
        

imgs = np.load('../data/img_rcc_' + str(256)+'.npy')
        
img_1 = imgs[100,0,:,:,:]
img_2 = imgs[100,1,:,:,:]

similarity = calculate_similarity(img_1, img_2)
print(similarity)