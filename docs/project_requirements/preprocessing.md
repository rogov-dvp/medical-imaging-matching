# Preprocessing

Documentation on the preprocessing aspect of the code base.

This section of code does the following:

- cleans the data
- resizes the image
- saves the image in the correct file type and location

## Detailed Description

Our preprocessing section of the code deals with formatting and editing the images/mammograms we are analyzing. This ensures that the images we feed the model are of the size and file type we expect. Specifically, this section of code will clean and remove noisy data, resize the images (we are currently working with 256x256 images), and save the image in the correct location (according to the type of mammogram) in the expected file type (NumPy array). We are also able to augment the images via zooming, flipping, and other methods. This is helpful for us as it allows us to artificially grow our sample size and therefore increase the accuracy of our model.

Data cleaning is the first step in our preprocessing script. Our code will iterate through the mammograms, clean them, remove any duplicates, handle missing data, and flags numeric and non-numeric columns. We then move to preprocess_data which is the main chunk of the preprocessing code. This script will iterate through the stated location of the mammograms. It will then check to see if the mammogram has already been processed or not. If it has, it will return the location of the already processed mammogram, if it has not, it will continue with the preprocessing. The preprocessing consists of loading the image, resizing it, and then saving it in accordance with the mammogram type (ie. LCC, RCC, RMLO, LMLO) as a NumPy array.

The preprocessing section of our code is run any time the model is run. If we do need more images, we will run the data augmentation scripts which create altered images of the original mammograms. Specifically, it will create a zoomed image, a shifted image horizontally, a shifted image vertically, a brighter image, a vertically flipped image, a horizontally flipped image, and a rotated image.
