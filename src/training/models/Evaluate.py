import numpy as np

def calculate_avg_dist(network, imgs_1, imgs_2):
    embeddings_1 = network.predict(imgs_1)
    embeddings_2 = network.predict(imgs_2)
    avg_dist = np.mean(abs(embeddings_1 - embeddings_2))
    return avg_dist