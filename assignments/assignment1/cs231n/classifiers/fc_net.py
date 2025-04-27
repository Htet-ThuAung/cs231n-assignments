from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight_scale = weight_scale

    def loss(self, X, y=None):
        grads = {}

        # Check the parameters at the start
        print("W1 at the start of loss:", self.params['W1'])

        # Forward pass
        print("X shape:", X.shape)
        print("W1 shape:", self.params['W1'].shape)
        print("b1 shape:", self.params['b1'].shape)

        hidden_layer, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(hidden_layer, self.params['W2'], self.params['b2'])

        if y is None:
            return scores

        # Compute loss and gradients
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))

        # Backward pass
        dhidden, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dhidden, cache1)

        # Regularization gradient
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2']

        print("[DEBUG] Entering loss()")
        print("W1 is None?", self.params['W1'] is None)

        return loss, grads
