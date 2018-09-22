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
    'dim', 64, 'The dimensionality of hidden layer.')
tf.app.flags.DEFINE_integer(
    'z_dim', 126, 'The dimensionality of noise.')
tf.app.flags.DEFINE_integer(
    'epoch', 50, 'The number of epoch.')
tf.app.flags.DEFINE_integer(
    'disc_iters', 5, 'The number of iterations for discriminator.')
tf.app.flags.DEFINE_integer(
    'con_dim', 0, 'The index of variational conditional variable for test, 0 or 1.')

tf.app.flags.DEFINE_float(
    'learning_rate', 1e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'lambd', 1., 'The coefficient of similarity regularization.')
tf.app.flags.DEFINE_float(
    'gp_lambd', 10., 'The weight of gradient penalty.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.5, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.9, 'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_string(
    'data_dir', '/home/clb/dataset',
    'The directory to load dataset svhn.')
tf.app.flags.DEFINE_string(
    'result_dir', 'result',
    'The directory to save result of scgan.')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'checkpoint',
    'The directory to save or load checkpoint file.')
tf.app.flags.DEFINE_string(
    'dataset_type', 'svhn',
    'The dataset svhn.')


FLAGS = tf.app.flags.FLAGS
data_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset_type)
result_dir = os.path.join(FLAGS.result_dir, 'scgan_' + FLAGS.dataset_type + '_continuous')
checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, 'scgan_' + FLAGS.dataset_type + '_continuous')


def discriminator(input_x, is_training=True, reuse=False):
    with tf.variable_scope('discriminator', values=[input_x], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            biases_initializer=tf.constant_initializer(0.0),
                            activation_fn=leaky_relu,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = slim.conv2d(input_x, FLAGS.dim, [5, 5], stride=2, scope='d_conv1', normalizer_fn=None)
                    net = slim.conv2d(net, 2*FLAGS.dim, [5, 5], stride=2, scope='d_conv2')
                    net = slim.conv2d(net, 4*FLAGS.dim, [5, 5], stride=2, scope='d_conv3')
                    net = slim.flatten(net)
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
                    net = slim.fully_connected(input_z, 64*FLAGS.dim, scope='g_fc1')
                    net = tf.reshape(net, [-1, 4, 4, 4*FLAGS.dim])
                    net = slim.conv2d_transpose(net, 2*FLAGS.dim, [5, 5], stride=2, scope='g_dconv2')
                    net = slim.conv2d_transpose(net, FLAGS.dim, [5, 5], stride=2, scope='g_dconv3')
                    net = slim.conv2d_transpose(net, 3, [5, 5], stride=2, scope='g_dconv4', activation_fn=None)
                    return tf.nn.tanh(net), net



x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')
z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
c = tf.placeholder(tf.float32, [None, 2], name='c')
# c_one_hot = tf.one_hot(c, 10)
zc = tf.concat([z, c], axis=1)

# structure
D_real, D_real_logits = discriminator(x, is_training=True, reuse=False)
G_fake, G_fake_logits = generator(zc, is_training=True, reuse=False)
D_fake, D_fake_logits = discriminator(G_fake, is_training=True, reuse=True)

# loss for discriminator
d_loss = tf.reduce_mean(D_fake_logits - D_real_logits)


# gradient penalty
alpha = tf.random_uniform([FLAGS.batch_size, 1, 1, 1], 0., 1.)
interpolates = alpha*x+(1.-alpha)*G_fake
_, D_inter_logits = discriminator(interpolates, is_training=True, reuse=True)
grad = tf.gradients(D_inter_logits, [interpolates])[0]
grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1,2,3]))
gradient_penalty = tf.reduce_mean(tf.square(grad_norm-1.))
d_loss += FLAGS.gp_lambd * gradient_penalty


# loss for generator
g_loss = tf.reduce_mean(-D_fake_logits)


if FLAGS.train:
    # similarity regularization
    sim_reg_list = []
    for i in range(FLAGS.batch_size):
        for j in range(i+1, FLAGS.batch_size):
            a = 1. - tf.abs(c[i]-c[j])
            sim = tf.sqrt(tf.reduce_sum(tf.square(G_fake[i]-G_fake[j])))
            sim_reg_list.append(tf.reduce_sum(a*sim+(1.-a)/(sim+1e-5)))
    sim_reg = tf.truediv(tf.add_n(sim_reg_list), FLAGS.batch_size*(FLAGS.batch_size-1.))
    g_loss += FLAGS.lambd * sim_reg


# trainable variable
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.op.name]
g_vars = [var for var in t_vars if 'generator' in var.op.name]

# optimizers
d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator') + \
                tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator_1')
with tf.control_dependencies(d_update_ops):
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.adam_beta1,
                                    beta2=FLAGS.adam_beta2).minimize(d_loss, var_list=d_vars)
with tf.control_dependencies(g_update_ops):
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.adam_beta1,
                                    beta2=FLAGS.adam_beta2).minimize(g_loss, var_list=g_vars)


# test
fake_images, _ = generator(zc, is_training=False, reuse=True)

# dataset
trainX, trainY = utils.load_svhn(data_dir)
train_gen = utils.get_batch(trainX, trainY, FLAGS.batch_size)

# for test
sample_z = np.random.normal(size=[FLAGS.batch_size, FLAGS.z_dim])
sample_c = np.random.uniform(0., 1., size=[FLAGS.batch_size, 2])

saver = tf.train.Saver(max_to_keep=20)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def train():
    num_batches = FLAGS.epoch * (len(trainX) // FLAGS.batch_size)
    d_loss_list, g_loss_list, sim_reg_list, step_list = [], [], [], []
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(num_batches):
            for j in range(FLAGS.disc_iters):
                batch_x = (next(train_gen)[0]/255.-.5)*2.
                batch_z = np.random.normal(size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
                batch_c = np.random.uniform(0., 1., size=[FLAGS.batch_size, 2])
                    
                # update discriminator
                _, _d_loss = sess.run([d_optim, d_loss], feed_dict={x:batch_x, z:batch_z, c:batch_c})

            # update generator
            batch_z = np.random.normal(size=[FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
            batch_c = np.random.uniform(0., 1., size=[FLAGS.batch_size, 2])
            _, _g_loss, _sim_reg = sess.run([g_optim, g_loss, sim_reg], feed_dict={z:batch_z, c:batch_c})

            if i % 10 == 0:
                print('Step: [%5d], time: %4.2f, d_loss: %.8f, g_loss: %.8f, sim_loss: %.8f' \
                    % (i+1, time.time() - start_time, _d_loss, _g_loss, _sim_reg))
                start_time = time.time()

            if i % 30 == 0:
                d_loss_list.append(_d_loss)
                g_loss_list.append(_g_loss)
                sim_reg_list.append(_sim_reg)
                step_list.append(i)


            if i % 1000 == 999:
                utils.save_plot(step_list, g_loss_list, 
                                os.path.join(result_dir, 'g_loss.jpg'),
                                title_name='SCGAN on ' + FLAGS.dataset_type.upper(), y_label_name='g_loss')
                utils.save_plot(step_list, d_loss_list, 
                                os.path.join(result_dir, 'd_loss.jpg'),
                                title_name='SCGAN on ' + FLAGS.dataset_type.upper(), y_label_name='d_loss')
                utils.save_plot(step_list, sim_reg_list, 
                                os.path.join(result_dir ,'sim_reg.jpg'),
                                title_name='SCGAN on ' + FLAGS.dataset_type.upper(), y_label_name='sim_reg')
                

            if i % 300 == 299:
                samples, = sess.run([fake_images], feed_dict={z:sample_z, c:sample_c})
                utils.save_images(samples[:30][...,::-1], [3, 10], os.path.join(
                        result_dir, 'step_{}.png'.format(i+1)), norm='[-1,1]')
            
            if i % 10000 == 9999:
                saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=i+1)
        
        # the last batch
        saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=num_batches)


def test():
    assert 0<=FLAGS.con_dim<=1, 'The con_dim must be 0 or 1.'
    lins = np.linspace(0., 1., 10)
    image_list = []
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    # ckpt = 'E:/python_workspace/paper-scgan-copy/checkpoint/scgan_svhn_continuous/model.ckpt-100000'
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        test_z = np.random.normal(size=[32, FLAGS.z_dim])
        test_c = np.zeros([32, 2])
        test_c[:, 1-FLAGS.con_dim] = 0.5 + np.random.normal(0.0, 0.001, 32)
        for i in range(10):
            test_c[:, FLAGS.con_dim] = lins[i] + np.random.normal(0.0, 0.001, 32)
            samples, = sess.run([fake_images], feed_dict={z:test_z, c:test_c})
            image_list.append(samples.reshape([32, -1]))
    images = np.concatenate(image_list, axis=0).reshape((10, 32, 32, 32, 3))
    images = np.transpose(images, [1,0,2,3,4])
    images = images.reshape((-1, 32, 32, 3))
    utils.save_images(images[...,::-1], [32, 10], 'svhn_c%d.png' % FLAGS.con_dim, norm='[-1,1]')
    np.save('svhn_c%d.npy' % FLAGS.con_dim, images)




def main(_):
    if FLAGS.train:
        train()
    else:
        test()


if __name__ == '__main__':
    tf.app.run()