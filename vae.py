import random
import os

import tensorflow as tf
from tensorflow.contrib import slim

class VAE():
    """A class representing Variational Autoencoder"""

    def __init__(self, input_dim, z_dim, sampling=True, scope='VAE'):

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.activation = tf.nn.relu
        self.sampling = sampling
        self.scope = scope

        
        self._create_graph()
        os.makedirs('summary', exist_ok=True)
        sub_d = len(os.listdir('summary'))
        self.train_writer = tf.summary.FileWriter(logdir = 'summary/'+str(sub_d))
        self.merged = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1000)

    # --------------------------------------------------------------------------
    def _create_graph(self):
        print('Create_graph')
        self.x, self. learning_rate, self.is_training =  self._input_graph()

        self.z, self.z_mu, self.z_log_sigma = self.encoder(self.x, sampling=self.sampling)
        tf.summary.histogram('Z hist', self.z)
        self.logits, self.x_ = self.decoder(self.z)

        self.loss = self.create_cost_graph(logits=self.logits, original=self.x,
            z_mu=self.z_mu, z_log_sigma=self.z_log_sigma)

        self.train_step = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)
        print('Done!')

    # --------------------------------------------------------------------------
    def _input_graph(self):
        print('\t_input_graph')

        x = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        learning_rate = tf.placeholder(tf.float32, ())
        is_training = tf.placeholder(tf.bool, ())

        return x, learning_rate, is_training

    # --------------------------------------------------------------------------
    def encoder(self, x, sampling=True, reuse=False):
        print('\tencoder')

        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder', reuse=reuse):
                x = tf.reshape(x, (-1, 28, 28, 1))
                regularizer = slim.l2_regularizer(0.001)
                # encoder
                net = slim.conv2d(
                                  x,
                                  32,
                                  [3, 3],
                                  activation_fn=self.activation,
                                  weights_regularizer=regularizer)
                net = slim.max_pool2d(net, [2, 2], stride=2)
                net = slim.batch_norm(
                                      net,
                                      scale=True,
                                      updates_collections=None,
                                      is_training=self.is_training)
                net = slim.conv2d(
                                  net,
                                  64,
                                  [3, 3],
                                  activation_fn=self.activation,
                                  weights_regularizer=regularizer)
                net = slim.max_pool2d(net, [2, 2], stride=2)
                net = slim.flatten(net)
                net = slim.fully_connected(
                                            net,
                                            2 * self.z_dim,
                                            activation_fn=None,
                                            weights_regularizer=regularizer)
                #split the layer to mu and sigma
                z_mu, z_log_sigma = tf.split(net, 2, 1)

                if sampling:
                    z = self.GaussianSample(z_mu, tf.exp(z_log_sigma))
                else:
                    z = z_mu

        return z, z_mu, z_log_sigma

    # --------------------------------------------------------------------------
    def decoder(self, z, reuse=False):
        print('\tdecoder')

        with tf.variable_scope(self.scope):
            with tf.variable_scope('decoder', reuse=reuse):
                regularizer = slim.l2_regularizer(0.001)

                net = slim.fully_connected(
                                           z,
                                           7 * 7 * 64,
                                           activation_fn=self.activation,
                                           weights_regularizer=regularizer)
                net = tf.reshape(net, (-1, 7, 7, 64))
                net = slim.conv2d_transpose(
                                            net,
                                            32,
                                            [3, 3],
                                            stride=2,
                                            activation_fn=self.activation,
                                            weights_regularizer=regularizer)
                net = slim.batch_norm(
                                      net,
                                      scale=True,
                                      updates_collections=None,
                                      is_training=self.is_training)
                net = slim.conv2d_transpose(
                                            net,
                                            1,
                                            [3, 3],
                                            stride=2,
                                            activation_fn=self.activation,
                                            weights_regularizer=regularizer)
                net = slim.flatten(net)
                logits = slim.fully_connected(
                                              net,
                                              self.input_dim,
                                              activation_fn=None,
                                              weights_regularizer=regularizer)

        return logits, tf.nn.sigmoid(logits)

    # --------------------------------------------------------------------------
    def create_cost_graph(self, logits, original, z_mu, z_log_sigma):
        print('\tcreate_cost_graph')
        self.ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=original), 1)
        self.kl_loss = tf.reduce_sum(self.KL(z_mu, tf.exp(z_log_sigma)), 1)
        self.l2_loss = tf.add_n(tf.losses.get_regularization_losses())

        tf.summary.scalar('Cross entropy loss', tf.reduce_mean(self.ce_loss))
        tf.summary.scalar('L2 loss', self.l2_loss)
        tf.summary.scalar('KL loss', tf.reduce_mean(self.kl_loss))
        return tf.reduce_mean(self.ce_loss + self.kl_loss, 0) + self.l2_loss


    # --------------------------------------------------------------------------
    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config)
        return self._sess

    # --------------------------------------------------------------------------
    def train(self, batch_size, learning_rate, data_loader):

        for i in range(data_loader.train.num_examples//batch_size):
            x, _ = data_loader.train.next_batch(batch_size)
            feed_dict = {
                         self.x: x,
                         self.learning_rate: learning_rate,
                         self.is_training: True}
            _, summary = self.sess.run([self.train_step, self.merged],
                feed_dict=feed_dict)
            self.train_writer.add_summary(summary, i)


    # --------------------------------------------------------------------------
    def predict(self, data_loader):

        x = random.sample(list(data_loader.validation.images), 100)
        feed_dict = {
                     self.x: x,
                     self.is_training: False}
        ce, kl = self.sess.run([self.ce_loss, self.kl_loss], feed_dict=feed_dict)        
        print('Cross-entropy loss: {}, KL loss: {}'.format(ce.mean(), kl.mean()))

    # --------------------------------------------------------------------------
    def get_z(self, x):
        return self.sess.run(self.z, feed_dict={self.x: x,
                                                self.is_training: True})

    # --------------------------------------------------------------------------
    def reconstruct(self, x):
        return self.sess.run(self.x_, feed_dict={self.x: x,
                                                 self.is_training: True}) 

    # --------------------------------------------------------------------------
    def reconstruct_from_z(self, z):
        logits, x = self.sess.run(self.decoder(z, reuse=True), feed_dict={self.is_training: True})
        return x        

    # --------------------------------------------------------------------------
    def save_model(self, path, global_step=None):
        self.saver.save(self.sess, path, global_step=global_step)

    # --------------------------------------------------------------------------
    def load_model(self, path):
        self.saver.restore(self.sess, path)

    # --------------------------------------------------------------------------
    def KL(self, mu, sigma, mu_prior=0.0, sigma_prior=1.0, eps=1e-7):
        return -(1/2)*(1 + tf.log(eps + (sigma/sigma_prior)**2) \
            - (sigma**2 + (mu - mu_prior)**2)/sigma_prior**2)

    # --------------------------------------------------------------------------
    def GaussianSample(self, mu, sigma):
        return mu + sigma*tf.random_normal(tf.shape(mu), dtype=tf.float32)


