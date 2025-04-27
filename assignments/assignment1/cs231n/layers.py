from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_reshaped = x.reshape(x.shape[0], -1)
    print("x_reshaped_shape: ", x_reshaped.shape)
    print("w shape", w.shape)
    out = np.dot(x_reshaped, w) + b
    print("out shape: ", out.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_flattened = x.reshape(x.shape[0], -1)

    # Compute the gradients with respect to x
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)

    # Compute the gradients with respect to w
    dw = x_flattened.T.dot(dout)

    # Compute the gradients with respect to b
    db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Apply relu activation function
    out = np.maximum(0, x)

    # Cache the input for the backward bass
    cache = x

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)  # if  x> 0, dx=dout; otherwise dx=0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute the mean and variance of each feature
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        # Normalize the data
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

        # Apply the scale and shift
        out = gamma * x_hat + beta

        # Update the running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # Store the cache for backward pass
        cache = (x, x_hat, sample_mean, sample_var, gamma, beta, eps)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # Apply the scale and shift
        out = gamma * x_hat + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack the cache from forward pass
    x, x_hat, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape

    # Compute the gradients with respect to gamma and beta
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    # Backpropagate through the normalization step
    dx_hat = dout * gamma

    # Compute the gradient of the variance
    dvar = np.sum(
        dx_hat * (x - sample_mean) * -0.5 * (sample_var + eps) ** (-1.5), axis=0
    )

    # Compute the gradient of the mean
    dmean = (
        np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0)
        + dvar * np.sum(-2 * (x - sample_mean), axis=0) / N
    )

    # Compute the gradient with respect to the input x
    dx = (
        (dx_hat / np.sqrt(sample_var + eps))
        + (dvar * 2 * (x - sample_mean) / N)
        + (dmean / N)
    )
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack the cache from the forward pass
    x, x_hat, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape

    # Compute gradients for gamma and beta
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    # Compute the gradient of the input
    dx_hat = dout * gamma

    # Simplify gradient calculations for variance and mean
    dvar = np.sum(
        dx_hat * (x - sample_mean) * -0.5 * (sample_var + eps) ** (-1.5), axis=0
    )
    dmean = (
        np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0)
        + dvar * np.sum(-2 * (x - sample_mean), axis=0) / N
    )

    # Final gradient with respect to the input x
    dx = (
        (dx_hat / np.sqrt(sample_var + eps))
        + (dvar * 2 * (x - sample_mean) / N)
        + (dmean / N)
    )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the mean of variance of each data point
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)

    # Normalize the datapoints
    x_hat = (x - mean) / np.sqrt(var + eps)

    # Scale and shift using gamma and beta
    out = gamma * x_hat + beta

    # Cache intermediate values for the backward pass
    cache = (x, x_hat, mean, var, gamma, beta, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve the cached values
    x, x_hat, mean, var, gamma, beta, eps = cache

    N, D = dout.shape

    # Gradient with respect to beta
    dbeta = np.sum(dout, axis=0)

    # Gradient with respect to gamma
    dgamma = np.sum(dout * x_hat, axis=0)

    # Gradient with respect to x_hat
    dx_hat = dout * gamma

    # Compute the gradient of x
    dvar = np.sum(
        dx_hat * (x - mean) * -0.5 * (var + eps) ** (-3 / 2), axis=1, keepdims=True
    )
    dmean = (
        np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=1, keepdims=True)
        + dvar * np.sum(-2 * (x - mean), axis=1, keepdims=True) / D
    )

    # Compute the gradient of x
    dx = dx_hat / np.sqrt(var + eps) + 2 * dvar * (x - mean) / D + dmean / D

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Training mode: apply dropout
        mask = np.random.rand(*x.shape) < p  # Create the dropout mask
        out = x * mask  # Apply the mask to the inputt
        out /= p  # Scale the output by 1/p to maintain expected value

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Test model: no dropout, return the input as is
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask
        dx /= dropout_param["p"]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract convolution parameters
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    # Get the dimensions of the input and filters
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Compute the dimensions of the output
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Pad with input x
    x_padded = np.pad(
        x, ((0,), (0,), (pad,), (pad,)), mode="constant", constant_values=0
    )

    # Initialize the output
    out = np.zeros((N, F, H_out, W_out))

    # Perform the convolution
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    # Find the receptive field in the input
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the region of interest
                    x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # Compute the convolution for this position
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + [f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve the necessary variables from the cache
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    # Get the dimension of the input, weights and output
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, F, H_out, W_out = dout.shape

    # Initialize gradients with respect to x, w, and b
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Pad the input x to account for padding
    x_padded = np.pad(
        x, ((0,), (0,), (pad,), (pad,)), mode="constant", constant_values=0
    )

    # Compute the gradients with respect to the weights and biases
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the region of interest
                    x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # Compute the gradient with respect to filter weights
                    dw[f] += x_slice * dout[n, f, i, j]

                    # Compute the gradient with respect to the biases
                    db[f] += dout[n, f, i, j]

                    # Compute the gradient with respect to the input
                    dx_padded = dout[n, f, i, j] * w[f]
                    dx[n, :, h_start:h_end, w_start:w_end] += dx_padded

    if pad > 0:
        dx = dx[:, :, pad:-pad, pad:-pad]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve parameters
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    # Get the dimensions of the input
    N, C, H, W = x.shape

    # Compute the output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # Initialize output
    out = np.zeros((N, C, H_out, W_out))

    # Perform max-pooling
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # Compute the starting and ending indices for the pooling window
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Perform max-pooling by taking the maximum value in the window
                    out[n, c, i, j] = np.max(x[n, c, h_start:h_end, w_start:w_end])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve the input and pool parameters from the cache
    x, pool_param = cache
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    # Get the dimensions of the input and the output
    N, C, H, W = x.shape
    N, C, H_out, W_out = dout.shape

    # Initialize dx to be the same shape as x
    dx = np.zeros_like(x)

    # Loop through the batch, channels, output height, and output width
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    # Compute the starting and ending indices for the pooling window
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Find the max value in the current pooling window
                    pool_region = x[n, c, h_start:h_end, w_start:w_end]
                    max_value = np.max(pool_region)

                    # Find the position of the max value in the pooling region
                    max_index = np.unravel_index(
                        np.argmax(pool_region), pool_region.shape
                    )

                    # The gradient of the input x is only updated at the position of the max value
                    dx[n, c, h_start:h_end, w_start:w_end][max_index] += dout[
                        n, c, i, j
                    ]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mode = bn_param["mode"]
    eps = bn_param["eps", 1e-5]
    momentum = bn_param.get("momentum", 0.9)
    running_mean = bn_param.get("running_mean", np.zeros((x.shape[1],), dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros((x.shape[1],), dtype=x.dtype))

    N, C, H, W = x.shape

    # Reshape the input x
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)

    # Perform batch normalization on reshaped input
    if mode == "train":
        mean = np.mean(x_reshaped, axis=0)
        var = np.var(x_reshaped, axis=0)

        # Normalize the data
        x_normalized = (x_reshaped - mean) / np.sqrt(var + eps)

        # Scale and shift
        out_reshaped = gamma * x_normalized + beta

        # Update the running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        # Cache for backpropagation
        cache = (x_reshaped, x_normalized, mean, var, gamma, beta, eps)

    elif mode == "test":
        # Normalize using running mean and variance
        x_normalized = (x_reshaped - running_mean) / np.sqrt(running_var + eps)
        out_reshaped = gamma * x_normalized + beta

    else:
        raise ValueError("Invalid forward mode: ", mode)

    # Reshape output back to the original shape
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # Update the bn_param with running mean and variance
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_reshaped, x_normalized, mean, var, gamma, beta, eps = cache
    N, C, H, W = dout.shape

    # Reshape dout to match the shape of x_shaped
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)

    # Bachprop through the scaling and shifting
    dgamma = np.sum(dout_reshaped * x_normalized, axis=0)
    dbeta = np.sum(dout_reshaped, axis=0)

    # Backprob through normalization
    dx_normalized = dout_reshaped * gamma

    # Compute gradients of variance and mean
    dvar = np.sum(
        dx_normalized * (x_reshaped - mean) * -0.5 * np.power(var + eps, -1.5), axis=0
    )
    dmean = np.sum(dx_normalized * -1 / np.sqrt(var + eps), axis=0) + dvar * np.mean(
        -2 * (x_reshaped - mean), axis=0
    )

    # Backprob through the input
    dx_reshaped = (
        dx_normalized * 1 / np.sqrt(var + eps)
        + dvar * 2 * (x_reshaped - mean) / N
        + dmean / N
    )

    # Reshape dx back to the original shape (N, C, H, W)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get the dimensions of the input
    N, C, H, W = x.shape

    # Reshape x into (N*G, C//G * H * W) for easier computation
    x_reshaped = x.reshape(N * G, C // G * H * W)

    # Compute mean and variance over the groups
    mean = np.mean(x_reshaped, axis=1, keepdims=True)
    var = np.var(x_reshaped, axis=1, keepdims=True)

    # Normalize the data
    x_normalized = (x_reshaped - mean) / np.sqrt(var + eps)

    # Reshape the normalized data to original shape
    x_normalized = x_normalized.reshape(N, C, H, W)

    # Apply the scale and shift parameters
    out = gamma * x_normalized + beta

    # Cache the necessary values for the backward pass
    cache = (x, gamma, beta, G, x_normalized, mean, var, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve the cache
    x, gamma, beta, G, x_normalized, mean, var, eps = cache

    N, C, H, W = x.shape

    # Reshape dout to have the same shape as the normalized x
    dout_reshaped = dout.reshape(N * G, C // G * H * W)

    # Compute the gradients for gamma and beta
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3), keepdims=True)

    # Compute gradient of normalized x (dx_normalized)
    dx_normalized = dout * gamma

    # Reshape dx_normalized back to its original shape
    dx_normalized = dx_normalized.reshape(N * G, C // G * H * W)

    # Compute the derivative of the normalization step
    var_eps = var + eps
    dvar = np.sum(
        dx_normalized * (x - mean) * -0.5 * np.power(var_eps, -1.5),
        axis=1,
        keepdims=True,
    )
    dmean = np.sum(
        dx_normalized * -1 / np.sqrt(var_eps), axis=1, keepdims=True
    ) + dvar * np.sum(-2 * (x - mean), axis=1, keepdims=True) / (C // G * H * W)

    # Compute the gradient with respect to x
    dx = (
        (dx_normalized / np.sqrt(var + eps))
        + (dvar * 2 * (x - mean) / (C // G * H * W))
        + (dmean / (C // G * H * W))
    )

    # Reshape dx to have the same shape as the original input x
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C = x.shape

    # Compute the correct class scores
    correct_class_scores = x[np.arange(N), y]

    # Compute the margins for all classes (except for the correct class)
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1)
    margins[np.arange(N), y] = 0  # Do not count the correct class

    # Compute the loss
    loss = np.sum(margins) / N

    # Compute the gradient
    dx = np.zeros_like(x)
    margins[margins > 0] = 1  # Set the positive margin to 1

    # Assign margins to dx
    dx = margins

    # Scale the gradient by 1 / N
    dx /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C = x.shape

    # Shift the input scores to prevent overflow during exponentiation
    x_shifted = x - np.max(x, axis=1, keepdims=True)

    # Compute the class probabilities using softmax
    exp_scores = np.exp(x_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute the loss: average negative log likelihood
    correct_class_probs = probs[np.arange(N), y]
    loss = -np.sum(np.log(correct_class_probs)) / N

    # Compute the gradient
    dx = probs
    dx[np.arange(N), y] -= 1  # Subtract 1 for the correct class
    dx /= N  # Average over the batch

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
