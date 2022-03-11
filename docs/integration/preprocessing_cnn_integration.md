# Integration

## Preprocessing and CNN Model Integration

In order for our system to work correctly, the CNN Model and the preprocessing scripts must work together.

### Current State

Right now the CNN Model and preprocessing scripts are working separately and are not integrated.

There are a few different ways we can approach a fix for this.

### Proposals

The different proposals depend on how we are testing and training the model. Our end goal is to have a script ie. `main.py` that will take in mismatched, check if the data is preprocessed (if not send it to preprocessing), then use that data to detect if the two images are from the same patient or not.

However, we also must train our model so our repository is split into different folders for training and prediction. The `main.py` script is within prediction and not connected to our model currently as it is not ready for prediction.

1. We can take `main.py` and copy it to the training folder. Then, we will add in the connection to the CNN model.

2. The other option is to call on the `preprocessing` scripts from within the CNN model.

Once the scripts are connected, we then need to ensure that we are calling upon the images that result from the `preprocessing`.

### Current Issues

1. On the BC Cancer remote server, we are unable to run anything that requires `open-cv`. I have checked that `open-cv` is installed on both `conda` and `pip` however it does not seem to recognize it. Everytime we attempt to import `import cv2` we get the error `ImportError: No module cv2`. I think we need to perhaps have Quinn take a look with elevated permissions.
