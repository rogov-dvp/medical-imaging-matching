import numpy as np

def validate_model(data, model):
    
    anchor = data[0]
    positive = data[1]
    negative = data[2]
    
    embed_a = model.predict(anchor)
    embed_p = model.predict(positive)
    embed_n = model.predict(negative)
    
    pos_distance = sum(np.sqrt(np.sum(np.square(embed_a - embed_p),axis=1))) / anchor.shape[0]
    neg_distance = sum(np.sqrt(np.sum(np.square(embed_a - embed_n),axis=1))) / anchor.shape[0]
    
    return pos_distance, neg_distance