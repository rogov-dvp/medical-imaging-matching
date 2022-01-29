# Batch Builder
- Exploration into implementing batch building methods in order to return not only loss but all embeddings. Finding images from previous batches and picking ones with most similarity.

- Training the CNN model involves giving a batch of samples to the model. For an arbitrary sample, one can obtain the embedding vector.

## Embeddings
- Taking input and creating a smaller representation such as a vector
- Similar images produce small distance between the two
- Different images produce a large distance between them

## Implementation
- Match negative nased on output of last learning step 
- Need to return all embedding not only loss 
- Determine by looking at where the anchor is compared to negative and positive
- Triplet loss function implemented using Keras
- Triplets need to be good in order for training

## Training
- Each loop builds a new batch in order to process
- Mant strategies possible that all have various impacts on the training speeds  

## Sources 
-  https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
-  https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973
- https://www.researchgate.net/figure/Batch-construction-of-triplet-loss-and-N-pair-mc-loss_fig2_333418402
-  file:///Users/marie/Downloads/sensors-21-00764-v2.pdf