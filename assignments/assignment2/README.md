### CS231n's Assignment2

This repository contains my solutions and experiments from Assignment 2 of CS231n. The assignment deepened my understanding of neural networks and introduced key techniques used in training deep models - from batch normalization and dropout to convolutional layers and PyTorch. Below is a summary of what I worked in each notebook.


`FullyConnectedNets.ipynb` - Multi-Layer Neural Network
I implmented a flexible class to build fully connected networks of any depth. This include writing my own backpagation logic and experimenting with various update strategies like SGD with momentum, RMSProp, and Adam. I tuned network depth, learning rate, and regularization to improve accuracy.

`BatchNormalization.ipynb` - Batch Norm
Implemented the forward and backward passes for batch normalization from scratch. I then integrated it into my deep network and ran experiments to observe its effects on training stability and speed. This was one of the most insightful parts of the assignment.

`Dropout.ipynb` - Regularization with Dropout
Added dropout to the fully connected network to help prevent overfitting. I ran compraisons with and without dropout and analyzed how it impacted generalization, especially on smaller training sets.

`ConvolutionalNetworks.ipynb` - CNN Layers
Built several essential components of a convolutional network including conv, max-pooling, and spatial batchnorm layers. These were combined into a modular CNN which I trained on CIFAR-10. This gave me a much clearer picture of how CNNs work under the hood.

`PyTorch.ipynb` - Deep Learning with PyTorch
Trained a CNN from scratch using the framework. I designed a small architecture and tuned it to reach solid performance on CIFAR-10. Working in PyTorch gave me exporsure to industry-standard tooling and workflows.