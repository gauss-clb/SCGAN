import tensorflow as tf
import utils
import numpy as np
import os
import time
from ops import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

slim = tf.contrib.slim

tf.app.flags.DEFINE_boolean(
    'train', True, 'Whether to train or test.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'z_dim', 90, 'The dimensionality of noise.')
tf.app.flags.DEFINE_integer(
    'epoch', 25, 'The number of epoch.')

tf.app.flags.DEFINE_float(
    'learning_rate', 0.0002, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.5, 'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_string(
    'data_dir', '/home/clb/dataset',
    'The directory to load dataset mnist or fashion-mnist.')
tf.app.flags.DEFINE_string(
    'result_dir', 'result',
    'The directory to save result of cgan.')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'checkpoint',
    'The directory to save or load checkpoint file.')
tf.app.flags.DEFINE_string(
    'dataset_type', 'mnist',
    'mnist or fashion-mnist')


FLAGS = tf.app.flags.FLAGS
data_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset_type)
result_dir = os.path.join(FLAGS.result_dir, 'cgan_' + FLAGS.dataset_type)
checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, 'cgan_' + FLAGS.dataset_type)


def discriminator(input_x, input_c, is_training=True, reuse=False):
    with tf.variable_scope('discriminator', values=[input_x, input_c], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            biases_initializer=tf.constant_initializer(0.0),
                            activation_fn=leaky_relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = slim.conv2d(input_x, 64, [4, 4], stride=2, scope='d_conv1', normalizer_fn=None)
                    net = slim.conv2d(net, 128, [4, 4], stride=2, scope='d_conv2')
                    net = slim.flatten(net)
                    net = tf.concat([net, input_c], axis=1)
                    net = slim.fully_connected(net, 1024, scope='d_fc3')
                    net = slim.fully_connected(net, 1, scope='d_fc4', normalizer_fn=None, activation_fn=None)
                    return tf.nn.sigmoid(net), net



def generator(input_z, is_training=True, reuse=False):
    with tf.variable_scope('generator', values=[input_z], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            biases_initializer=tf.constant_initializer(0.0),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = slim.fully_connected(input_z, 1024, scope='g_fc1')
                    net = slim.fully_connected(net, 128*7*7, scope='g_fc2')
                    net = tf.reshape(net, [-1, 7, 7, 128])
                    net = slim.conv2d_transpose(net, 64, [4, 4], stride=2, scope='g_dconv3')
                    net = slim.conv2d_transpose(net, 1, [4, 4], stride=2, scope='g_dconv4', activation_fn=None)
                    return tf.nn.sigmoid(net), net


x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
c = tf.placeholder(tf.int32, [None], name='c')
c_one_hot = tf.one_hot(c, 10)
zc = tf.concat([z, c_one_hot], axis=1)

# structure
D_real, D_real_logits = discriminator(x, zc, is_training=True, reuse=False)
G_fake, G_fake_logits = generator(zc, is_training=True, reuse=False)
D_fake, D_fake_logits = discriminator(G_fake, zc, is_training=True, reuse=True)

# loss for discriminator
d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real))
d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake))
d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

# loss for generator
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

# trainable variable
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.op.name]
g_vars = [var for var in t_vars if 'generator' in var.op.name]

# optimizers
d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator') + \
                tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator_1')
with tf.control_dependencies(d_update_ops):
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.adam_beta1).minimize(d_loss, var_list=d_vars)
with tf.control_dependencies(g_update_ops):
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate*5, beta1=FLAGS.adam_beta1).minimize(g_loss, var_list=g_vars)

# test
fake_images, _ = generator(zc, is_training=False, reuse=True)

# dataset
trainX, trainY, testX, testY = utils.load_mnist(data_dir)
train_gen = utils.get_batch(trainX, trainY, FLAGS.batch_size)

# for test
sample_z = np.random.uniform(-1., 1., [FLAGS.batch_size, FLAGS.z_dim])
sample_c = (np.arange(FLAGS.batch_size) % 10).astype(np.int32)

saver = tf.train.Saver()
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def train():
    num_batches = FLAGS.epoch * (len(trainX) // FLAGS.batch_size)
    d_loss_list, g_loss_list, step_list = [], [], []
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(num_batches):
            batch_x, batch_c = next(train_gen)
            batch_x = batch_x / 255.
            batch_c = batch_c.astype(np.int32)
            batch_z = np.random.uniform(-1., 1., [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
                

            # update discriminator
            _, _d_loss = sess.run([d_optim, d_loss], feed_dict={x:batch_x, z:batch_z, c:batch_c})
            # update generator
            _, _g_loss = sess.run([g_optim, g_loss], feed_dict={z:batch_z, c:batch_c})

            if i % 10 == 0:
                print('Step: [%5d], time: %4.2f, d_loss: %.8f, g_loss: %.8f' \
                    % (i+1, time.time() - start_time, _d_loss, _g_loss))
                start_time = time.time()

            if i % 30 == 0:
                d_loss_list.append(_d_loss)
                g_loss_list.append(_g_loss)
                step_list.append(i)


            if i % 1000 == 999:
                utils.save_plot(step_list, g_loss_list, 
                                os.path.join(result_dir, 'g_loss.jpg'),
                                title_name='CGAN on ' + FLAGS.dataset_type.upper(), y_label_name='g_loss')
                utils.save_plot(step_list, d_loss_list, 
                                os.path.join(result_dir, 'd_loss.jpg'),
                                title_name='CGAN on ' + FLAGS.dataset_type.upper(), y_label_name='d_loss')
                

            if i % 300 == 299:
                samples, = sess.run([fake_images], feed_dict={z:sample_z, c:sample_c})
                utils.save_images(samples[:30], [3, 10], os.path.join(
                        result_dir, 'step_{}.png'.format(i+1)), norm='[0,1]')
            
            if i % 10000 == 9999:
                saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=i+1)
        
        # the last batch
        saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=num_batches)


def test():
    image_list = []
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        for i in range(313):
            test_z = np.random.uniform(-1., 1., [FLAGS.batch_size, FLAGS.z_dim])
            test_c = np.random.randint(0, 10, size=[FLAGS.batch_size])
            samples, = sess.run([fake_images], feed_dict={z:test_z, c:test_c})
            image_list.append(samples.reshape([FLAGS.batch_size, -1]))
    images = np.concatenate(image_list, axis=0)
    np.save(os.path.join(result_dir , 'cgan_' + FLAGS.dataset_type + '.npy'), 
            (images[:10000]*255.).astype(np.uint8))




def main(_):
    if FLAGS.train:
        train()
    else:
        test()


if __name__ == '__main__':
    tf.app.run()