**Models**

- We are planning to try a Convolutional Neural Net with 4 Convolutional and 4 pooling layers, connected to a Neural Net first, to experiment with TensorFlow and start things off. We are not expecting really good results from this, but think that it is a good approach to set everything up and see whether every component of the system works before we start trying more complex models and experiment with the other components.

| **Option** | **info** |
|-----------|-----------|
| Euclidean Distance| <ul><li>Measure the ‘similarity’ between two vectors</li><li>Use when calculating the distance between two rows of data containing numerical values, for example, two pixels</li></ul>|
| RootSIFT| <ul><li>Used to describe and detect features within digital images, through locating key points and descriptors, often used for object recognition</li></ul>|
|Siamese Networks| <ul><li>Finds similarity through comparison of feature vectors</li><li>Very Powerful networks used in facial recognition</li></ul>|
|MSE/SSIM| <ul><li>Simple technique used in order to find similarity of two images</li><li>MSE will calculate mean square error (not highly indicative of similarity), SSIM will look for the similarity in the pixels of two images</li></ul>|
| Convolutional Neural Net| <ul><li>Widely used in the field of Computer Vision, to analyse images and condense information out of them</li><li>Easy to build with eg. tensorflow</li></ul>|
| FaceNET| <ul><li>Lots of documentation, and great performance</li><li>Learns face-specific features especially in deeper layers, which are not applicable for us</li></ul>|