### CS231n's Assignment1

This repository includes my implementation of various classifiers from scratch as part of CS231n's assignment. The assignment introduced me to foundational techniques in image classification, from basic distance metrics to training a simple neural network. Each part helped build up my understanding of how models learn from data.


`knn.ipynb` - k-Nearest Neighbor Classifier
Implemented a basic k-NN classifier using fully vectorized NumPy code. This exercise helped sharpen my skills in writing efficient, reliable logic without relying on libraries like scikit-learn.

`svm.ipynb` - Linear Support Vector Machine
Built a multiclass SVM classifier and manually derived the loss and gradients. Implementing SGD from scratch gave me hands-on insight into how optimization is handled in real-world training loops.

`softmax.ipynb` - Softmax with Cross-Entropy
Extended my SVM work to the Softmax classifier. By adjusting the loss function and interpreting the output as probabilities, I saw how small changes in formulation inpact model behavior and performance.

`two_layer_net.ipynb` - Neural Network
Developed a two-layer fully connected neural network with ReLU activations and trained it using gradient descent. I tuned hyperparameters like learning rate and regularization strength to improve validation accuracy.

`features.ipynb` - Higher-level Representations
Explored feature extraction techniques (e.g, HOG, color histograms) to boost classification performance over raw pixels. This showed me the importance of feature engineering, especially when using simpler models.
