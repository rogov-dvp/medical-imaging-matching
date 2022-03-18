import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reshape_to_128(triplets):
    batch_size = triplets[0].shape[0]
    out=[np.zeros((batch_size, 128, 128, 3)) for i in range(3)]
    
    for i in range(3):
        for j in range(batch_size):
            myarray = (triplets[i][j,:,:,:] * 255).astype(np.uint8)
            img = Image.fromarray(myarray)
            img = img.resize(size=(128,128))
            img = np.asarray(img)
            out[i][j,:,:,:] = img / 255
            
    return out