# Preprocessing

Documentation on the preprocessing aspect of the code base.

This section of code does the following:

- cleans the data
- resizes the image
- saves the image in the correct file type and location

We also have code that will augment images if needed for the training of our model.

This code is run automatically when the model is being ran.

If the image has already been processed, it will use the already processed image. If it has not, it will process the image.

See code for more info.
