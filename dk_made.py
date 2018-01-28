import numpy as np
import tensorflow as tf
from functools import reduce


def _get_rank(max_rank, num_out, rng=np.random):
    rank_out = np.array([])
    while len(rank_out) <= num_out:
        rank_out = np.concatenate([rank_out, np.arange(max_rank)])
    excess = len(rank_out) - num_out
    remove_ind = rng.choice(max_rank, excess, False)
    rank_out = np.delete(rank_out, remove_ind)
    rng.shuffle(rank_out)
    return rank_out.astype('float32')


def _get_mask_from_ranks(r1, r2):
    return (r2[:, None] >= r1[None, :]).astype('float32')


def get_masks_all(ds, rng=np.random):
    # ds: list of dimensions dx, d1, d2, ... dh, dx,
    #                       (2 in/output + h hidden layers)
    dx = ds[0]
    ms = list()
    rx = _get_rank(dx, dx, rng=rng)
    r1 = rx
    for d in ds[1:-1]:
        r2 = _get_rank(dx - 1, d, rng=rng)
        ms.append(_get_mask_from_ranks(r1, r2))
        r1 = r2
    r2 = rx - 1
    ms.append(_get_mask_from_ranks(r1, r2))
    assert np.all(np.diag(reduce(np.dot, ms[::-1])) == 0), 'wrong masks'

    return ms


def get_tf_masks(layers_sizes, scope=None, rng=np.random):
    with tf.variable_scope(scope, "made_masks"):
        tf_masks = []
        for i, mask in enumerate(get_masks_all(layers_sizes, rng=rng)):
            tf_masks.append(tf.get_variable('mask%d'%i, initializer=mask.T, trainable=False))
    return tf_masks


#def masked_dense_layer(x, mask, weight_initializer=None, bias_initializer=tf.zeros_initializer(), scope=None):
def masked_dense_layer(x, mask, weight_initializer=None, bias_initializer=tf.constant_initializer(0.), scope=None):
    """Simple dense layer with masked weights."""
    with tf.variable_scope(scope, "masked_dense", [x]):
        w = tf.get_variable('w', mask.shape, tf.float32, initializer=weight_initializer)
        b = tf.get_variable('b', mask.shape[1], tf.float32, initializer=bias_initializer)
        return tf.matmul(x, w * mask) + b

def made(x, context, hidden_layer_sizes, repeat_last_layer, activation_fn, scope=None):

    with tf.variable_scope(scope, "made"):
        n_inputs = x.get_shape().as_list()[1]
        masks = get_tf_masks((n_inputs,) + tuple(hidden_layer_sizes) + (n_inputs,))

        # Reshape to tile and the last mask and obtain many output variables.
        last_mask = tf.reshape(masks[-1], masks[-1].shape.concatenate((1,)))
        last_mask = tf.tile(last_mask, (1, 1, repeat_last_layer))
        masks[-1] = tf.reshape(last_mask, (last_mask.shape.as_list()[0], -1))

        h = x
        for i, mask in enumerate(masks):
            h = masked_dense_layer(h, mask, scope="made%d" % i)
            if i < len(masks) - 1:
                if context is not None:
                    h += tf.contrib.layers.linear(context, h.shape.as_list()[1], scope="context%d" % i)
                h = activation_fn(h)

        h = tf.reshape(h, (-1, n_inputs, repeat_last_layer))
        return h

# maintaining pixelSNAIL API
def dk_made(x, h, init, ema, dropout_p,
            nr_resnet, nr_filters, attn_rep, att_downsample, resnet_nonlinearity, 
            n_out):
    return made(x, None, [nr_filters,] * nr_resnet, n_out, resnet_nonlinearity)

