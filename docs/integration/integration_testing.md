# Integration

## Integration Testing

Integration testing exercises two or more parts of an application at once, including the interactions between the parts, to determine if they function as intended. This type of testing identifies defects in the interfaces between disparate parts of a codebase as they invoke each other and pass data between themselves.

This will mostly be between preprocessing and the model.

### How is integration testing different from unit testing?

While unit testing is used to find bugs in individual functions, integration testing tests the system as a whole. These two approaches should be used together, instead of doing just one approach over the other.

Integration tests operate on a higher level of abstraction than unit tests. The main difference between integration and unit testing is that integration tests actually affect external dependencies.

Example:

|Unit Tests|Integration Tests|
|:---------|-----------------|
|Unit testing works on component by component basis and hence if any small component to be tested, we need to go for unit testing.|Integration testing works on the whole component, we can conclude as many unit tests make integration testing. And also a software is complete if whole integration testing is complete.|
|Testing of addition calculation alone in a calculator program for specific set of arguments.|Testing of all arithmetic operations like addition, subtraction etc., (whatever present in calculator program) with different set of arguments.|

### How to Write Integration Tests?

We will use the `unittest` library but instead of focusing on small things, we will be focusing on scripts as whole or the integration between two methods.

### Areas that need Integration Testing

- Between preprocessing and cnn.
- Between main and everything that is called on in main.
- Test preprocessing and the saving of images. Ensure that the images will be where they are expected to be when called upon from a different script.
