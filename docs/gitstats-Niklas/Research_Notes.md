# Notes on Research Papers

## ChexNet
[https://arxiv.org/abs/1711.05225](https://arxiv.org/abs/1711.05225)

- Network Trained on XRays, initial thought that structure they detect is way more similar to what is learned in ImageNet, maybe we can start transfer learning form here
- Model can be found [here](https://github.com/brucechou1983/CheXNet-Keras)
- Also started training from ImageNet, with Adam Beta_1 = 0.9 and Beta_2=0.999 Learning Rate=0.001 and decay of factor 10 afer plateau


## One Shot Learning
[https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

- Weights initialised with zero mean and std=0.01
- Biases initialised with mean=0.5 and std=0.01
- Training Parameters similar to CheXNet

Interesting for us to vary if we want to experiment with learning from scratch:
- Output layer size (they experimented with sizes between 128 and 4096)
- Change convolution size (3x3,5x5,...,20x20)

- Use the Omingolet Datasetfor testing, this is the Benchmark for few shot learning and see whether our model progresses on this dataset
