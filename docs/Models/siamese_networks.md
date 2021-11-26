# Siamese Networks 
Created: November 19th, 2021

Last Updated: November 19thth, 2021

## Description
A Siamese network is a type of neural network that consists of many identical networks. These networks receive a pair of inputs. The features of one input are computed by each network. Then the dot product or difference of the features is used to calculate their similarity. The goal output is 1 for input pairs from the same class, and 0 for input pairs from other classes.
Keep in mind that the parameters and weights for both networks are identical. They aren't Siamese if they aren't.

### Libraries 
Possible libraries to consider in the process:
 1. Tensorflow
 2. NumPy
 3. Keras Library

### Techniques 
1. Building pairs 
2. Training 
3. Euclidean distance to determine similarity
4. Triplet loss  


### Implementation 
Focus on preparing the data,creating layers, then creating the model, using implementation triplet loss function, and then testing and training the model.

### References 
- https://keras.io/examples/vision/siamese_network/
- https://www.pyimagesearch.com/2020/12/07/comparing-images-for-similarity-using-siamese-networks-keras-and-tensorflow/
- https://towardsdatascience.com/siamese-networks-introduction-and-implementation-2140e3443dee
- https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463
