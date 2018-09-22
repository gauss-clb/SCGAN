import tensorflow as tf
import numpy as np

batch_norm_decay=0.997
batch_norm_epsilon=1e-5
batch_norm_scale=True


batch_norm_params = {
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
}


def leaky_relu(x):
    return tf.maximum(x, 0.2*x)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)


def resize_image(image,
                 new_height=32,
                 new_width=32,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('ResizeImage',
                        values=[image, new_height, new_width, method, align_corners]):
        new_image = tf.image.resize_images(image, 
                                           [new_height, new_width],
                                           method=method,
                                           align_corners=align_corners)
        new_image.set_shape([new_height, new_width, 3])
        return new_image