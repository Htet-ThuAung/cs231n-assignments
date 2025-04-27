from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_reshaped = x.reshape(N, D)

    # Compute the output
    out = np.dot(x_reshaped, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Reshape x to (N, D)
    x_reshaped = x.reshape(x.shape[0], -1)

    # Compute the gradients
    # Gradient of the loss with respect to x
    dx = np.dot(dout, w.T).reshape(x.shape)

    # Gradient of the loss with respect to w
    dw = np.dot(x_reshaped.T, dout)

    # Gradient of the loss with respect to b
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0] # number of samples
    C = x.shape[1] # number of classes

    # Compute the softmax values (probabilities)
    shift_x = x -  np.max(x, axis=1, keepdims=True)
    #print(f"Input to softmax (x):\n{x}") # debug
    #print(f"Shifted scores (for numerical stability):\n{shift_x}") # debug print

    softmax_probs = np.exp(shift_x) / np.sum(np.exp(shift_x), axis=1, keepdims=True)
    #print(f"Softmax probabilities:\n{softmax_probs}") # debug print

    # Compute the loss
    correct_logprobs = -np.log(softmax_probs[np.arange(N), y])
    loss = np.sum(correct_logprobs) / N # average loss over all samples

    # Debugging: Check the label indices and softmax probabilities
    #print(f"Indices for softmax_probs: {np.arange(N)}, {y}")

    # Compute the gradient with respect to x
    softmax_probs[np.arange(N), y] -= 1 # Subtract 1 from correct classes
    dx = softmax_probs / N # gradient of the softmax loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

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

    # Debugging prints
    # print(f"x.shape: {x.shape}")
    # print(f"gamma.shape: {gamma.shape}")
    # print(f"beta.shape: {beta.shape}")
    # print(f"running_mean.shape: {bn_param.get('running_mean', np.zeros(D)).shape}")
    # print(f"running_var.shape: {bn_param.get('running_var', np.zeros(D)).shape}")



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

        # Compute mean and variance of the current batch
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        #print("Mean shape:", mean.shape)
        #print("Variance shape:", var.shape)
        # Normalize the input x
        x_normalized = (x - mean) / np.sqrt(var + eps)

        # Scale and shift using gamma and beta
        out = gamma * x_normalized + beta

        #print("Variance before adding eps:", var)
        #print("Variance after adding eps:", var + eps)

        # Update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        #print("Updated running_mean:", running_mean)
        #print("Updated running_var:", running_var)
        #print("Batch normalization mode:", mode)

        
        # Cache the values
        cache = (x, x_normalized, mean, var, gamma, beta, eps)
        #print("Cache assigned:", cache)

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

        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)

        # Scale and shift using gamma and beta
        out = gamma * x_normalized + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    #print("Output shape:", out.shape)
    #print("Cache contents:", cache)
    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

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

    # Retrieve cache value
    x, x_normlaized, mean, var, gamma, beta, eps = cache
    N, D = x.shape

    # Gradient with respect to scale (dgamma) and shift (dbeta)
    dgamma = np.sum(dout * x_normlaized, axis=0)
    dbeta = np.sum(dout, axis=0)

    # Gradient with respect to normalized x
    dnormalized_x = dout * gamma

    # Gradient with respect to variance
    dvariance = np.sum(dnormalized_x * (x - mean) * -0.5 * (var + eps) ** -1.5, axis=0)

    # Gradient with respect to mean (dmean)
    dmean = np.sum(dnormalized_x * -1 / np.sqrt(var + eps), axis=0) + dvariance * np.sum(-2 * (x - mean), axis=0) / N

    # Gradient with respect to x (dx)
    dx = dnormalized_x / np.sqrt(var + eps) + dvariance * 2 * (x - mean) / N + dmean / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

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

    # Retrieve cache values
    x, x_normalized, mean, var, gamma, beta, eps = cache
    N, D = x.shape

    # Gradient with respect to scale (dgamma) and shift (dbeta)
    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)

    # Gradient with respect to x (dx)
    dx = (dout * gamma) / np.sqrt(var + eps) - np.mean(dout * gamma, axis=0) / np.sqrt(var + eps) - \
         np.mean(dout * gamma * (x - mean) * 2 / (var + eps), axis=0) / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

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

    # Compute mean and variance of each data point
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)

    # Normalize the input data 
    x_normalized = (x - mean) / np.sqrt(var + eps)

    # Scale and shift the normalized data using gamma and beta
    out = gamma * x_normalized + beta

    # Cache the intermediated variables needed for backward pass
    cache = (x, x_normalized, mean, var, gamma, beta, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

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

    # Unpack cache
    x, x_normalized, mean, var, gamma, beta, eps = cache
    N, D = dout.shape

    # Compute gradients for gamma and beta
    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)

    # Compute gradients for normalized input
    dx_normalized = dout * gamma

    # Compute gradient for x using chain rule
    std_dev = np.sqrt(var + eps)

    # Gradient with respect to mean and variance
    dvar = np.sum(dx_normalized * (x - mean) * -0.5 * np.power(std_dev, -3), axis=1, keepdims=True)
    dmean = np.sum(dx_normalized * -1 / std_dev, axis=1, keepdims=True) + dvar * np.sum(-2 * (x - mean), axis=1, keepdims=True) / N

    # Compute final gradient
    dx = (dx_normalized / std_dev) + (dvar * 2 * (x - mean) / N) + (dmean / N)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

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

        # Create a mask where each value is 1 with probability p, and 0
        # with probability (1 - p)
        mask = (np.random.randn(*x.shape) < p) / p # Scale the mask by 1/p
        out = x * mask # Apply the mask to the input


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x # No dropout in test phase

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

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

        dx = dout * mask # Apply the mask to the upstream derivatives

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

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

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    stride = conv_param['stride']
    pad = conv_param['pad']

    # Pad the input with zeros on all sides
    x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    #print(f"x shape before padding: {x.shape}")
    #print(f"x_padded shape after padding: {x_padded.shape}")
    # Output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Initialize the output array
    out = np.zeros((N, F, H_out, W_out))

    # Perform the convolution
    for i in range(N): # iterate over each input in the batch
        for f in range(F): # iterate over each filter
            for h in range(H_out): # iterate over output height
                for out_w in range(W_out): # iterate over output width
                    h_start = h * stride
                    h_end = h_start + HH
                    w_start = out_w * stride
                    w_end = w_start + WW

                    # Print the relevant information to debug
                    #print(f"Processing batch {i}, filter {f}, output position ({h}, {w})")
                    #print(f"x_padded[{i}, :, {h_start}:{h_end}, {w_start}:{w_end}] shape: {x_padded[i, :, h_start:h_end, w_start:w_end].shape}")
                    #print(f"w[{f}, :, :, :] shape: {w[f, :, :, :].shape}")

                    # Print shapes for debugging
                    # print(f"w type: {type(w)}")
                    # print(f"w shape: {w.shape}")
                    # print(f"x_padded[{i}, :, {h_start}:{h_end}, {w_start}:{w_end}].shape: {x_padded[i, :, h_start:h_end, w_start:w_end].shape}")
                    # print(f"w[{f}, :, :, :].shape: {w[f, :, :, :].shape}")

                    # Apply the filter to the receptive field
                    out[i, f, h, out_w] = np.sum(x_padded[i, :, h_start:h_end, w_start:w_end] * w[f, :, :, :]) + b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

# import numpy as np

# # Example data and filter initialization
# x = np.random.randn(2, 3, 4, 4)  # 2 images, 3 channels, 4x4 height and width
# w = np.random.randn(3, 3, 3, 3)  # 3 filters, 3 channels, 3x3 filter size
# b = np.linspace(-0.1, 0.2, num=3)  # Biases for the 3 filters
# conv_param = {'stride': 2, 'pad': 1}

# # Run convolution
# out, cache = conv_forward_naive(x, w, b, conv_param)
# print(out.shape)



def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

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

    x, w, b, conv_param = cache
    N, C, H, W = x.shape  # N: batch size, C: channels, H: height, W: width
    F, C, HH, WW = w.shape  # F: number of filters, C: channels, HH: height of filter, WW: width of filter
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # Padding input x for the backward pass
    x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    
    # Output dimensions from the forward pass
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    # Initialize gradients
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Compute db: sum dout over the spatial dimensions (H', W') for each filter
    db = np.sum(dout, axis=(0, 2, 3))  # Shape (F,)
    
    # Compute dw and dx
    for i in range(N):  # Iterate over each input in the batch
        for f in range(F):  # Iterate over each filter
            for h in range(H_out):  # Iterate over output height
                for out_w in range(W_out):  # Iterate over output width
                    # Calculate the receptive field in the input
                    h_start = int(h * stride)
                    h_end = int(h_start + HH)
                    w_start = int(out_w * stride)
                    w_end = int(w_start + WW)
                    
                    # Compute gradient for the filter
                    dw[f, :, :, :] += x_padded[i, :, h_start:h_end, w_start:w_end] * dout[i, f, h, out_w]

                    # Compute gradient for the input
                    dx_padded[i, :, h_start:h_end, w_start:w_end] += w[f, :, :, :] * dout[i, f, h, out_w]
    # Remove padding from the gradient w.r.t. input
    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_padded
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

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

    N, C, H, W = x.shape  # N: batch size, C: channels, H: height, W: width
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    # Compute output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    # Initialize the output tensor with zeros
    out = np.zeros((N, C, H_out, W_out))
    
    # Perform max pooling
    for i in range(N):  # Iterate over the batch
        for c in range(C):  # Iterate over the channels
            for h in range(H_out):  # Iterate over the output height
                for w in range(W_out):  # Iterate over the output width
                    # Compute the start and end positions of the current pooling window
                    h_start = h * stride
                    h_end = h_start + pool_height
                    w_start = w * stride
                    w_end = w_start + pool_width
                    
                    # Extract the pooling region from the input
                    pool_region = x[i, c, h_start:h_end, w_start:w_end]
                    
                    # Take the maximum value in the pooling region
                    out[i, c, h, w] = np.max(pool_region)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

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

    x, pool_param = cache
    N, C, H, W = x.shape  # N: batch size, C: channels, H: height, W: width
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    # Compute output dimensions
    H_out = dout.shape[2]
    W_out = dout.shape[3]
    
    # Initialize the gradient with respect to x
    dx = np.zeros_like(x)
    
    # Perform the backward pass
    for i in range(N):  # Iterate over the batch
        for c in range(C):  # Iterate over the channels
            for h in range(H_out):  # Iterate over the output height
                for w in range(W_out):  # Iterate over the output width
                    # Compute the start and end positions of the current pooling window
                    h_start = h * stride
                    h_end = h_start + pool_height
                    w_start = w * stride
                    w_end = w_start + pool_width
                    
                    # Extract the pooling region from the input
                    pool_region = x[i, c, h_start:h_end, w_start:w_end]
                    
                    # Find the position of the maximum value in the pooling region
                    max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                    
                    # The gradient of dout is propagated to the location of the maximum value
                    dx[i, c, h_start + max_pos[0], w_start + max_pos[1]] += dout[i, c, h, w]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

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

    # Reshape input to (N * H * W, C) to apply batch normalization per channel
    N, C, H, W = x.shape
    #print("Shape of input x in forward pass: ", x.shape)
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)

    # Call the vanilla batch normalization function
    #print("Cache contents:", cache)
    out, batchnorm_cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    # Save the reshaped input and its original shape in cache
    cache = (x, batchnorm_cache)
    # Reshape the output back to (N, C, H, W)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

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

    # # Get the input data and reshape it to (N * H * W, C)
    # x, gamma, beta, mean, variance, running_mean, running_var = cache
    # print("Shape of x:--", x.shape)
    # N, C, H, W = x.shape
    # dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    # print("dout_reshaped shape:", dout_reshaped.shape)  # Debug print
    
    # # Call the vanilla batchnorm_backward function
    # dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    # print("dx_reshaped shape before reshaping:", dx_reshaped.shape)

    # # Reshape dx back to the shape (N, C, H, W)
    # dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    # print("dx shape after reshaping:", dx.shape)


    ####################
    x, batchnorm_cache = cache
    N, C, H, W = x.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)

    # Call the vanilla batchnorm_backward function
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, batchnorm_cache)

    # Reshape dx back to the shape (N, C, H, W)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

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

    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    N, C, H, W = x.shape

    # Reshape x into (N, G, C // G, H, W)
    x_grouped = x.reshape(N, G, C // G, H, W)
    
    # Compute mean and variance for each group
    mean = x_grouped.mean(axis=(2, 3, 4), keepdims=True)
    variance = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)
    
    # Normalize the data
    x_normalized = (x_grouped - mean) / np.sqrt(variance + eps)
    
    # Reshape normalized data back to (N, C, H, W)
    x_normalized = x_normalized.reshape(N, C, H, W)
    
    # Apply scale and shift
    out = gamma * x_normalized + beta
    
    # Store values needed for backward pass
    cache = (x, x_normalized, gamma, beta, G, mean, variance, eps)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

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

    # Unpack cache values
    x, x_normalized, gamma, beta, G, mean, variance, eps = cache
    N, C, H, W = dout.shape
    
    # Reshape dout to match the grouped structure: (N, G, C // G, H, W)
    dout_grouped = dout.reshape(N, G, C // G, H, W)
    
    # Reshape x_normalized to match the shape of dout_grouped: (N, G, C // G, H, W)
    x_normalized_grouped = x_normalized.reshape(N, G, C // G, H, W)

    # Compute dbeta and dgamma
    dbeta = dout_grouped.sum(axis=(0, 2, 3, 4), keepdims=True)  # (1, G, 1, 1)
    dgamma = (dout_grouped * x_normalized_grouped).sum(axis=(0, 2, 3, 4), keepdims=True)  # (1, G, 1, 1)

    # Compute the gradient of the normalized data
    dx_normalized = dout_grouped * gamma.reshape(1, G, C // G, 1, 1)  # Scale by gamma

    # Group the input data (x) to calculate group-wise mean and variance
    x_grouped = x.reshape(N, G, C // G, H, W)
    mean_grouped = x_grouped.mean(axis=(2, 3, 4), keepdims=True)  # Compute mean per group

    # Compute the gradient of variance and mean
    dvar = (dx_normalized * (x_grouped - mean_grouped) * -0.5 * (variance + eps) ** (-1.5)).sum(axis=(1, 2, 3), keepdims=True)
    dmean = (dx_normalized * -1 / np.sqrt(variance + eps)).sum(axis=(1, 2, 3), keepdims=True) + \
            dvar * (-2 * (x_grouped - mean_grouped)).sum(axis=(1, 2, 3), keepdims=True) / (N * H * W)

    # Compute the gradient of x
    dx = dx_normalized / np.sqrt(variance + eps) + dvar * 2 * (x_grouped - mean_grouped) / (N * H * W) + dmean / (N * H * W)

    # Reshape dx back to original input shape
    dx = dx.reshape(N, C, H, W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
