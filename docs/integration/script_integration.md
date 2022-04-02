# Medical Image Matching - Integration

How all the files and scripts flow within the Medical Image Matching System.

## File Structure

The medical image matching system has three main folders that it utilizes.

- `preprocessing`
- `src`
- folder containing the images
- `data`, which contains a csv file with a list of DICOM mismatches.

### `preprocessing`

This folder contains all the scripts used to prepare and clean the data.

- `data_cleaning.py` cleans our image data so that there is no noisy data etc.
- `data_cleaning_test.py` tests all the functionality within the `data_cleaning` file and ensures it is working as expected.
- `preprocess_data.py` takes the files from `data_cleaning`, performs any additional processes in order to make the image into what the model expects and then places the image in the correct location in the correct file type. For example, if the mammogram was of the RCC type, it would read the image, resize it, save it into the RCC folder within the preprocessed folder as a NumPy array. After which it can then be used for the model.
- `preprocess_data_remote.py` does the same as the `preprocess_data` file but instead of images it reads DICOM files and it is adjusted to reflect the file structure of the remote server.
- `preprocess_img.py` is the same as the `preprocess_data` file, however, it saves the files as an image not a NumPy array.
- `preprocess_test.py` this file contains the unittests for all the preprocessing files.

### `src`

This folder contains the main code of our matching system. Within this folder there are two subfolders, `training` and `predicting`.

#### `training`

This folder contains the source code to the image matching models we are running in order to determine if images are a match.

- `models` folder
  - all of our previous model(s) including ones we are no longer using.
    - this also includes all the testing files for these models.
  - `read_preprocessed.py` reads the preprocessed images for the model.
  - `BatchBuilder.py` is code that will build batches for our Triplet Loss Function and predicton model.
  - `MainRead.py` is the file that controls all the reading for our model.
  - `MainTrain.py` is the controlling file for training our model.
  - `MainTrainCNN.py` is the file that is used to specifically train the CNN.
  - `CNNTripletModel.py` this file contains all the code that builds our CNN model.
- `data_augmentation.py` augments our testing data to create more images for us to test on.
- `da_numpy.py` will augment the data to create more images for us to test on and save them as NumPy arrays.

#### `predicting`

This folder contains the code that functions as a central hub for all the code within the system

- `main.py` is the code that primarily controls the flow and intregration of the system.
- `tests_main.py` this file will serve as the unittests for main and as the central hub for integration testing.
- `status_bar.py` will create a status bar within command-line to provide some user-feedback and keep the user up to date on the progress of the scripts.

### Image Folder

As of now, we have the `test_images_kaggle` folder that contains all the images we are using to help us develop and build the model. In the future this will be changes to reflect the data given to us by BC Cancer.

- `augmented_images` is the folder containing all images that have been augmented.
- `images` is the folder that contains all the original images.
- `processed_images` is the folder that contains all the processed images that we mentioned before.

## Integration

**Note:** Before you can run the MIM model you must have sufficient data and clean data. If you do not, run `data_augmentation.py` and `data_cleaning.py`.

How do all these pieces connect and work together?

The steps of the system are as follows:

1. Read in `csv` file that details mismatches.
2. Check if the two image files detailed in the `csv` are processed already.
   - If the image is already processed, use the preprocessed file.
   - If the image is not already preprocessed, run `preprocess_data`.
   - At some point we need to crop the images with the Breast Detection Algorithm
3. Now that all the images are preprocessed, we want to start the model with the `MainTrain.py` file.
   - This file will then call `MainRead` which will rely on `read_preprocessing`.
   - Now we must build our batches with `BatchBuilder.py`
   - Once all the images have been read and batches built, we will move onto training the model with `MainTrainCNN` which relies on `CNNTripletModel`.
   - Now that our model is trained, we can run it on the images in question from the `CSV` file.
4. Return the similarity percentage found by the CNN.

Due to some issues that have been discovered, we may not be able to integrate our model completely.
