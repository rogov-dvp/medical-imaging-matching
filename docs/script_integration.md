# Medical Image Matching - Integration

How all the files and scripts flow within the Medical Image Matching System.

## File Structure

The medical image matching system has three main folders that it utilizes.

- `preprocessing`
- `src`
- folder containing the images

### `preprocesing`

This folder contains all the scripts used to prepare and clean the data.

- `data_augmentation.py` augments our testing data to create more images for us to test on.
- `data_cleaning.py` cleans our image data so that there is no noisy data etc.
- `preprocess_data.py` takes the files from `data_augmentation` and `data_cleaning`, performs any additional processes in order to make the image into what the model expects and then places the image in the `processed_images` folder (see Image Folder for more information).

### `src`

This folder contains the main code of our matching system. Within this folder there are two subfolders, `ims` and `models`.

#### `ims`

This folder contains the code that functions as a central hub for all the code within the system

- `main.py` is the code that primarily controls the flow and intregration of the system.
- `unprocessed.txt` is a file that lists all the cleaned and augmented data that has not been processed yet.
- `status_bar.py` will create a status bar within command-line to provide some user-feedback and keep the user up to date on the progress of the scripts.

#### `models`

This folder contains the source code to the image matching models we are running in order to determine if images are a match.

- `siamese.py` and `siamese2.py` are code for two different versions of a `siamese` model.
- `ssim.py` is the code for a SSIM model.

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
3. Once both images are processed, we pass those images to the `models`.
4. Return the similarity percentage found by the `models`.

The following table details which files perform each step:

|Step|File(s)|
|:---|:----|
|Step 1| `main.py`|
|Step 2| `main.py`|
|Step 2b| `preprocess_data.py`|
|Step 3| `models`|
|Step 4| `main.py`|
