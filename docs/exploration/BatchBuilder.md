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

## Storing the embeddings 
-  using call back function or should seconding be added in order to build dictionary to keep track of embedding??
-  From what the different implementations I have tried using the call back function to keep track of embedings is the best method

## Sources 
-  https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
- https://books.google.ca/books?id=sKXIDwAAQBAJ&pg=PA543&lpg=PA543&dq=example+of+batch+builder+in+python&source=bl&ots=V9GloWWFGn&sig=ACfU3U1wDyQMf7zXoZE6QedZtB906arvHQ&hl=en&sa=X&ved=2ahUKEwisyt3p-PH1AhXBGDQIHYQDBQQQ6AF6BAgREAM#v=onepage&q=example%20of%20batch%20builder%20in%20python&f=false
-  https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973
- https://www.researchgate.net/figure/Batch-construction-of-triplet-loss-and-N-pair-mc-loss_fig2_333418402
-  file:///Users/marie/Downloads/sensors-21-00764-v2.pdf