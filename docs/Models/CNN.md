# CNN Triplet Loss Documentation

## Packages needed
To run the model keras and (pydot) need to be installed.

## Model
### Nov 2021
The Model is for now a realtively simple CNN with three convolutional layers and small input size as well as small embedding vectors. These dimension are flexible and can be adjusted once we get to the actual problem. For now it has been mainly used with the MNIST data set to see whether it actually works.

### Jan 2022
The Model has been adjust in its implementation of the Triplet Loss functionality with Siamese Networks (that share the weights and structure). The Adjustments were required to be able to feed the loss back into the model for training the network. The adjustments are based on [Keras](https://keras.io/examples/vision/siamese_network/).

## Cost Function
The Model uses the triplet loss function, here the model calculates the embeddings for three input images, the anchor, a positive and negative sample. In order to get the loss the distances between the embeddings is calculated, such that anchor and negative get further apart and anchor and positive get closer together. In the next step the construction of the triplets can be optimzed as the  selection has great impact on learning progress.

## Predicting similarity 
When predicting the similarity of two images, one has to manually feed the model with the two images and calculate the distance of the output embeddings manually. This step still needs to be automized.

## Decision for this structure
The decision to continue on with this approach, is based on the positive impression we got during experimenting with the model. It gave us a really good modularity, so we can easily adjust the CNN the model is based on and by this can implement transfer learning, profiting from networks trained on millions of images. We also got encourage by the fact, that this architecture has shown great performance in other projects where the intent was to identify faces. This is nearly the same use case, just that there it were photos of faces that were tried to match, instead of mamograms.
