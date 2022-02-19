# CNN Model Research

CNN or Convolutional Neural Network is our base model of choice for detecting similarities between mammograms.

CNN utitlizes both traditional Deep Learning and Computer Vision techniques and builds on top of them to create something powerful and more importantly accurate.

## Convolutional Operation

The convolution operation is one of the foundations of the CNN. Mathematically speaking, a convolution is the integral measurng how much two functions overlap as one passes over the other (you can think of it as mixing two functions by multiplying them).

Typically, CNN's start with edge detection and as they develop the model starts to develop and be trained it can start detection partts of objects and eventually completed objects such as faces. For us, this is similar to the breast detection algorithm initially. We start with detection the edges of the breast in the mammogram and move forward by detecting more and more parts of the mammogram.

In terms of edge detection, the best approach for determining weights is to have the model learn them via backpropagation. By letting all of these numbers to be parameters and if we learn them automatically from data, we find that our new networks can actually learn low-level features. That is how we can learn features such as edges even more robustly than computer researchers are generally able to code up these things by hand.

Our CNN is most similar to FaceNet, in that is uses a deep convolutional network in conjunction with a triplet loss function.

## Embedding Issues - Triplet Loss Function

The CNN models we are using a Siamese Network in conjunction with a Triplet Loss Function to help our net learn similarities and differences between mammograms.

One way to learn the parameters of a neural network (so that we have a good encoding of the mammogram) is to define and apply a gradient descent on the Triplet Loss function.

As mentioned, a triplet loss function works by comparing an "anchor" image to a positive match and a negative match. We aim for the distance between the anchor and the positive to indicate similarity and the distance between negative and the anchor to indicate dissimaliriy.

Formally we want the parameters of our neural network, or our encodings, to have the following property (think of $d$ as a distance function):

$$||f(A)-f(P)||^2 \le ||f(A) - f(N)||^2$$
$$d(A, P) \le d(A, N)$$

or in a different form:

$$||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 \le 0$$

If $f$ always outputs 0, these two norms (distances) are $0-0=0$ and $0-0=0$, and by saying $f(img) = \bar{0}$, we can almost trivially satisfy this equation. We need to make sure that the neural network doesn't just output 0 for all encoding - that it doesn't set all the encodings equal to each other. One way for the neural network (NN) to give a trivial output is if the encoding for every image was identical to every other image, in which case we again get $0-0=0$. To prevent out network from doing that, we arre going to modify this objective so that it doesn't need to be just less than or equal to 0, it needss to be a bit smaller than 0. In particular, we say this needs to be less than $-\alpha$ where $\alpha$ is another hyper parameter (it is also called a margin).

$$|f(A) - f(P)|^2 - |f(A) - f(N)|^2 \le 0-\alpha$$

This looks like so after rearranging:

$$||f(A)-f(P)||^2 - ||f(A)-f(N)||^2 + \alpha \le 0$$

So, we want $d(A, N)$ to be much bigger than $d(A, P)$. To achieve this, we could either push $d(A, P)$ up or push $d(A, N)$ down, so that there is this gap of the hyperparameter $\alpha$ between the distance between the anchor and positive vs. the anchor and negative. This is the **role of a margin parameter**.

### Triplet Loss Function Definition

The triplet loss function is defined on triples of images as we mentioned before. The positive of examples of the same patient and the negatives are examples of a different patient. The loss will be defined as follows:

$$L(A, P, N) = max(||f(A)-f(P)||^2 - ||f(A)-f(N)||^2, 0)$$

As long as we achieve the goal of making $||f(A)-f(P)||^2 - ||f(A)-f(N)||^2$ less than or equal to zero. On the other hand, if this is greater than zero then we take the max so we get a positive loss.

This is how we define the loss on a single triplet, and the overall cost function for our NN can be a sum over a training set of these individual losses on different triplets:

$$J = \sum_{i = 1}^{M} h(A^{(i)}, P^{(i)}, N^{(i)})$$

Let’s imagine that we have a training set of 10,000 pictures with a 1000 different persons. Then, we have to take our 10,000 pictures and use them to generate triplets. Next, we train our learning algorithm using a gradient descent on cost function that we have defined previously. After which we can apply it to our problem.

#### **Choosing the triplets**

Choosing randomly while it could work, there is a high chance that $||f(A)-f(P)||^2 - ||f(A)-f(N)||^2$ will be much bigger than the margin $\alpha$ then that term on the left, and so the NN won't learn much. We actually want to form our triplets by choosing "hard cases" to train on.

In particular, we want all tripletes to satisfy this constraint ie. triplets where $d(A, P)$ and $d(A, N)$ are close.

$$d(A, P) + \alpha \le d(A, N)$$

$$d(A, P) \approx d(A, N)$$

In that case the learning algorithm has to try extra hard to take $d(A, N) and push it up. Choosing hard cases to train on increases the computational efficiency of our learning program. if we choose the triplets randomly then too many triplets would be really easy, and so gradient descent won’t do anything because neural network will just get them right pretty much all the time. So, with choosing hard triplets, the gradient descent procedure has to do some work to try to push $d(A, P)$ and $d(A, N)$ away from each other.

To train the loss function we need to use the training set of Anchor, Positive and Negative triples. Then, we will use gradient descent to try to minimize the cost function 𝐽 that we defined earlier. It will have the effect of back propagating to all the parameters of the neural network in order to learn an encoding. Hence, a function 𝑑 of two images will be small when these two images are of the same patient. However, they will be large when these are two images of different patients.

## Current Issues

As of now, we are mainly having issues with our loss dropping precipitously when it should not as well as forcing the model to look at the difference between mammograms intead of simply identifying them as mammograms.

### Loss Decline

As we found on the statistics stackoverflow, this problem has occurred in other uses of a triplet loss function, which is promising. The main suggestion to fix this issue is to read [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, Dmitry Kalenichenko, and James Philbin as it outlines many issues and solutions you may come across when using triplet loss functions.

However, they do provide additional details which we will go over below. For a review of the FaceNet article please see the `face_net_research` doc.

The triplet loss function aims to find an embedding such that:

$$||f(x_i^a)-f(x_i^p)||_2^2 + \alpha \lt ||f(x_i^a)-f(x_i^n)||_2^2 \forall (f(x_i^a), f(x_i^p), f(x_i^n)) \in T $$

where $T$ is the set of all possible triplets (which as we discussed above, is a set of an anchor, a positive and a negative).

Iterating over all possible triplets is expensive, especially with the size of our data. The loss of the above equation is 0 when the inequality holds and the loss becomes larger the more the inequality is violated. This gives us the loss function:

$$\begin{align*}
L = \sum_i max(0, ||f(x_i^a)-f(x_i^p)||_2^2 - ||f(x_i^a)-f(x_i^n)||_2^2 + \alpha)\\
= \sum_i ReLU (||f(x_i^a)-f(x_i^p)||_2^2 - ||f(x_i^a)-f(x_i^n)||_2^2 + \alpha)
\end{align*}
$$

*Note*: Not sure what $ReLU$ is referring to, it will require more research as it is not mentioned within this post

#### **Recommendations**

- Focus on the hardest triplets first, specifcially use *online hard-negative mining* to choose the triplets with the highest loss. See post to learn how to implement.
- Use large batch sizes.
- If loss gets stuck using *online hard-negative mining* start with *semi-hard* negative mining and switch to *hard-negative* mining later on.
- Watch for collapsed models, *semi-hard* negative mining can also help with this.

### Image Pipeline/Preprocessing

There may also be an issue with how we are feeding the images to the model. We are currently investigating this via a [Keras example/tutorial](https://keras.io/examples/vision/siamese_network/).

We suspect that there might be an issue due to not knowing precisely how the preprocessing done via VGG16 or ResNet50 works. Both of which we use in our model. There may also be an issue asa our images are greyscale whereas a lot of the images used in these models are RGB. Hopefully, our work in cropping and further research into how the images might affect our model will help.

## Sources

- [DataHacker #001 CNN Convolutional Neural Networks](https://datahacker.rs/computer-vision/)
- [DataHacker #002 CNN Edge Detection](https://datahacker.rs/edge-detection/)
- [DataHacker #003 CNN Edge Detection Extended](https://datahacker.rs/edge-detection-extended/)
- [DataHacker #032 CNN Triplet Loss](https://datahacker.rs/siamese-network-triplet-loss/)
- [Stats Stack Exchange - Triplet Loss Decline](https://stats.stackexchange.com/questions/475655/in-training-a-triplet-network-i-first-have-a-solid-drop-in-loss-but-eventually)
