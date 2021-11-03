# Data Preprocessing

Created: November 3rd, 2021

Last Updated: November 3rd, 2021

## Data Augmentation

As our data is fairly limited, we are going to be implementing data augmentation (DA) so we can train our models without risk of overfitting on the training dataset.

DA allows us to artificially expand the size of our training set by modifying our existing data. As our data consists of images, we will be focusing on utilizing image DA (though there are methods for audio, text, and other forms of data).

### DA Image Techniques

1. Geometric Transformations - randomly flipping, cropping, rotating, and/or translating images.
2. Color Space Transformations - change RGB color channels (intensify colors etc.)
3. Kernel filters - sharpen or blur an image
4. Random erasing - delete part of the initial image
5. Mixing images - mix an image with another

**Note:** Color space transformations will probably not be effective as our images are primarily greyscale.

#### **Image DA in Tensorflow**

- Write our own augmentation pipelines or layers using tf.image.

*If more DA/preprocessing techniques are needed we can utilize the `Keras` library, in particular the `ImageDataGenerator`*

#### **Other Image DA Libraries**:

- `PyTorch Transforms`
- `MxNet Transforms`
- `Augmentor`
- `Albumentations`
- `Imgaug`
- `AutoAugment (DeepAugment)`

### Implementation

The main methods we will be focusing on are Geometric Transformations, Kernel Filters, Random erasing, and mixing images.

### *References*

[Data Augmentation in Python](https://neptune.ai/blog/data-augmentation-in-python)
