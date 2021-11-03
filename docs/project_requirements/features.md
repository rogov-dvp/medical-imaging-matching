# Features


## Preprocessing
“The preprocessing must come up with a consistent set of images with same dimensions”
To feed the model with proper, consistent input we are going to implement several steps of preprocessing. This firstly includes data cleaning, with standardizing of the pixels to [0,1]. Secondly, we are going to resize the images with a 4x4 filter to reduce the size by 4, as the images have a high resolution as of now. Finally, we are going to apply Data Augmentation on the data set, to expand the database of available duplicates, as this is the limiting factor. Here we are considering rotations, shiftings, changing the brightness, and other augmenters.

## Data Organisation / Storage:
“Data must be organised, so the data is easily accessible for learning”
In the client meetings, we were informed that the current way the images are accessed might be a bit tedious for our use cases. Therefore we are planning to come up with an easier structure for our use case to store the preprocessed images.


## Test/Training Set Builder:
“Test/Training sets must represent the actual use case and support fast learning progress” 
As the final result highly depends on the learning process of the model and the learning highly depends on the data that are used for it, we are planning to invest some more time into this. Since we are comparing two images per mismatch the training and test dataset selection is incredibly important. We need to make sure we have enough mismatches and enough matches to train our model for both. If we have too many mismatches our model won't get used to matches and the processing of one might be biased or take too long. 
We are planning to set up positive examples where we are trying to find mammograms of the same person that are as different as possible and find negative examples that are similar to the one we are comparing it to. We are hoping that this improves the time it takes to train significantly.


## Models
“Models must return proper similarity scores and must be trained in reasonable time”
The model has to take two sets of mammograms or just two mammograms, as some sets are incomplete, and return a similarity score, to assess whether the mammograms are from the same patient. In order to come up with a similarity score, we are planning to test several image similarity algorithms - such as Euclidean Distance Model, RootSIFT, MSE/SSIM, and network models like Siamese Networks as well as other DNNs for image recognition. For the later, more complex models, we will be facing the problem that it will take a very long time and lots of data to train these networks from scratch. Therefore, we are planning to implement transfer learning, to profit from existing models that have already been trained on big data sets. The desired output of the networks would be an n-dimensional vector (n about 128) for each mammogram. These could then be compared with each other using euclidean distance or even another neural net to decide whether it is the same patient or not. To train this model, we would like to look into different cost functions like euclidean distance or triplet loss, based on euclidean distance, to speed up the learning process. In order to improve the performance of our model, we are also planning to optimize several hyperparameters like the learning rate, drop-out rate, momentum, batch size, and others.


