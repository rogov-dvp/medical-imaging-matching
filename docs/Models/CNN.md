# CNN Triplet Loss Documentation

## Packages needed
To run the model keras and (pydot) need to be installed.

## Model
The Model is for now a realtiv simple CNN with three convolutional layers and small input size as well as small embedding vectors. These dimension are flexible and can be adjusted once we get to the actual problem. For now it has been mainly used with the MNIST data set to see whether it actually works.

## Cost Function
The Model uses the triplet loss function, here the model calculates the embeddings for three input images, the anchor, a positive and negative sample. In order to get the loss the distances between the embeddings is calculate, such that anchor and negativ get further apart and anchor and positive get closer together. In the next step the cinstruction of the triplets can be optimzed as the  selection has great impact on learning progress.

## Predicting similarity 
When predicting the similarity of two images, one has to manually feed the model witht the two images and calculate the distance of the output embeddings manually. This step still needs to be automized.
