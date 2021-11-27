# Testing

## List of Testing to Implement

### Additional Unit Tests to Create:

#### Preprocessing

- `data_cleaning.py`
  - [ ] Add testing to ensure that we are reading the correct files.
  - [ ] Add testing to ensure that we are getting the output we are expecting when we read the files.
  - [ ] Ensuring that the split between non-numeric and numeric works.
  - [ ] Using dummy data, ensure that the missing values are found and dealt with.
- `preprocess_data.py`
  - [ ] Once this is completed, we will need to implement testing on the resizing of images, getting the images, getting the right images, and saving the now processed images.

#### src

- `models/cnn_tripletloss`
  - [ ] `buildDataSet` - test that it returns a list of length 10 containing the type of values that it should be.
  - [ ] `compute_dist` - test with dummy data that the calculation of distance is correct.
  - [ ] `get_batch_random_demo` - test that this method is returning a list containing 3 tensors A,P,N of shape (batch_size,w,h,c).
  - [ ] `get_batch_random` - test that it creates a batch of APN triplets with a completely random strategy
  - [ ] `opt_triplet` - ensure that this method returns the negative image in the type of variables we expected.
  - [ ] `build_network` - make sure that this method returns a neural net.
  - [ ] `TripletLossLayer` - make sure that when a TripletLossLayer object is made, it has the correct parameters and type.
    - [ ] `triplet_loss` - test that this returns the value and types that we expect.
    - [ ] `call` - test that this method returns what we expect.

#### training/data_augmentation

- [x] Test the augmentation methods.

#### training/models/ORB.py

- [ ] `orb_sim` - ensure that the method returns what we expect based on some dummy data.

#### training/models/SSIM.py

- [ ] `get_sim` - ensure that this method returns a value that makes sense in accordance with the dummy data and that it returns the right type of data.

#### training/models/siamese.py

- [ ] `make_pairs` - ensure that this method returns the values we expect based on dummy data and that it returns the right type of data.
- [ ] `euclidean_distance` - ensure that the result makes sense in accordance with the dummy data.
- [ ] `loss` - ensure that the return type is what we expect.

#### training/models/siamese2.py

- [ ] `preprocess_image` - test that it returns the file type we expect.
preprocess_triplets - ensure that the method returns what we expect.
- [ ] `euclid_distance` - ensure that the return type is what we expect and that based on the dummy data we pass in the calculation is correct.
- [ ] `DistanceLayer/call` - ensure that the return type is what is expected.
- [ ] `SiameseModel`
  - [ ] Ensure that all methods and attributes associated with this class return the type we expect and if applicable perform as expected with dummy data.
- [ ] `cosine_similarity` - ensure that the return type is what we expect and that based on the dummy data we pass in the calculation is correct.
