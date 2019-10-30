import tensorflow as tf


def conv_body(state, filters=(16,), kernel_sizes=(3,), strides=(3,), output_size=64):
    """
    Assumes the state shape is 3 dimensional. Applies some number of convolution layers, followed by a single
    dense layer.

    :param state: Tensor of shape [channels, batch, x_dim, y_dim]
    :param filters: Tuple of filters sizes.
    :param kernel_sizes: Tuple of kernel sizes.
    :param strides: Tuple of stride sizes. All tuples must be the same length.
    :param output_size: The number of units to output to.
    :return: A tensor of shape [batch_size, output_size]
    """
    architecture = list(zip(range(len(filters)), filters, kernel_sizes, strides))

    logits = tf.transpose(state, [0, 3, 2, 1])  # Transpose into channels last format
    with tf.variable_scope('ac_shared', reuse=tf.AUTO_REUSE):
        for i, filters, kernel_size, strides in architecture:
            logits = tf.layers.conv2d(logits, filters, kernel_size, strides, activation=tf.nn.leaky_relu,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      name='conv%d' % i)
        logits = tf.layers.flatten(logits)
        logits = tf.layers.dense(logits, units=output_size, activation=tf.nn.leaky_relu, name='d1')

    logits = tf.reshape(logits, [-1, output_size])
    return logits


def actor_spacial_head(features, sizes):
    """
    Feed forward network to calculate the spacial action probabilities.

    :param features: Tensor of shape [batch_size, num_features] inputs to the spacial_head
    :param sizes: List of integer sizes for spacial dimension sizes. Even indices are x dimensions and odd indices
        are y dimensions.

    :return:
        spacial_x: A list containing a tensor of size [batch_size, x_dimension] for each x dimension.
        spacial_y: A list containing a tensor of size [batch_size, y_dimension] for each y dimension.
    """
    spacial_x = []
    spacial_y = []
    with tf.variable_scope('actor_spacial_x', reuse=tf.AUTO_REUSE):
        for i in range(int(len(sizes) / 2)):
            spacial_x.append(tf.layers.dense(features,
                                             units=sizes[i * 2], activation=tf.nn.softmax, name='output%d' % i))
    with tf.variable_scope('actor_spacial_y', reuse=tf.AUTO_REUSE):
        for i in range(int(len(sizes) / 2)):
            spacial_y.append(tf.layers.dense(features,
                                             units=sizes[i * 2 + 1], activation=tf.nn.softmax, name='output%d' % i))
    return spacial_x, spacial_y


def actor_pointer_head(features, embeddings, num_heads):
    """
    Feed forward network that performs attention on on the embeddings using features `num_head` times.

    :param features: Tensor of shape `[batch_size, num_features]`
    :param embeddings: Tensor of shape `[batch_size, num_units, embedding_size]`
    :param num_heads: An integer representing the number of separate softmax distributions to output.

    :return: A softmax distribution over the units in the embedding of shape `[batch_size, num_heads, num_units]`
    """
    with tf.variable_scope('pointer_head', reuse=tf.AUTO_REUSE):
        hidden_size = 50
        mapped_features = tf.layers.dense(features, hidden_size, activation=None, use_bias=False)
        mapped_embeddings = tf.layers.dense(embeddings, hidden_size, activation=None, use_bias=False)

        mapped_features = tf.expand_dims(mapped_features, axis=1)
        logits = tf.layers.dense(tf.tanh(mapped_features + mapped_embeddings), num_heads,
                                 activation=None, use_bias=False)
        logits = tf.transpose(logits, [0, 2, 1])  # Now shape [batch_size, num_heads, num_units]

        # If embedding is all zeros, then this "unit" is padding and should be masked out
        is_real_unit_mask = tf.not_equal(tf.reduce_sum(embeddings, axis=-1), tf.constant(0, dtype=tf.float32))
        is_real_unit_mask = tf.expand_dims(is_real_unit_mask, axis=1)

        # TODO: Check math to make sure this is equivalent to not masking:
        probs = tf.nn.softmax(logits + 1e-10) * tf.cast(is_real_unit_mask, dtype=tf.float32) + 1e-10
        return probs / tf.reduce_sum(probs, axis=-1, keepdims=True)


def actor_nonspacial_head(features, action_mask, num_actions):
    """
    Feed forward network to produce the nonspacial action probabilities.

    :param features: Tensor of shape [batch_size, num_features]
    :param action_mask: Tensor of shape [batch_size, num_actions]
    :param num_actions: number of actions to produce.
    :return: Tensor of shape [batch_size, num_actions] of probability distributions.
    """
    with tf.variable_scope('actor_nonspacial', reuse=tf.AUTO_REUSE):
        probs = tf.layers.dense(features, units=num_actions, activation=tf.nn.softmax, name='output')
    masked = (probs + 1e-10) * action_mask
    return masked / tf.reduce_sum(masked, axis=1, keepdims=True)


def value_head(features):
    """
    Feed forward network to produce the state value.

    :param features: Tensor of shape [batch_size, num_features].
    :return: Tensor of shape [batch_size] of state values.
    """
    with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
        features = tf.layers.dense(features, units=1, activation=None, name='output')
    return tf.squeeze(features, axis=-1)