from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is not None:   # interpolation in DRAGAN
                shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
                alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
                inter = a + alpha * (b - a)
                inter.set_shape(a.shape)
            else:
                inter = a
            return inter
        x = _interpolate(real, fake)
        pred = f(x)
        grad = tf.gradients(pred, [x])[0]
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=-1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'Dirac':
        gp = _gradient_penalty(f, real, fake=None)
    elif mode == 'wgan_gp':
        gp = _gradient_penalty(f, real, fake)

    return gp



