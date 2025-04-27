from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    for i in range(N):
        scores = X[i].dot(W)
        # Shift scores to prevent numeric instability
        scores -= np.max(scores)
        # Compute the softmax probabilities
        softmax_scores = np.exp(scores) / np.sum(np.exp(scores))
        # Compute the loss for this example
        loss += -np.log(softmax_scores[y[i]])
        # Compute the gradient for this example
        for j in range(C):
            dW[:, j] += (softmax_scores[j] - (j == y[i])) * X[i]

  # Average loss and gradient over all examples
    loss /= N
    dW /= N

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    # Compute the raw scores (N x C)
    scores = X.dot(W)

    # Shift scores to prevent numeric instability
    scores -= np.max(scores, axis=1, keepdims=True)

    # Compute softmax scores (N x C)
    exp_scores = np.exp(scores)
    softmax_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute the loss using the correct class probabilities
    correct_class_scores = softmax_scores[range(N), y]
    loss = np.sum(np.log(correct_class_scores)) / N

    # Add regularization to the loss
    loss += 0.5 * reg * np.sum(W * W)

    # Compute the gradient with respect to W
    softmax_scores[range(N), y] -= 1
    dW = X.T.dot(softmax_scores) / N

    # Add regularization to the gradient
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
