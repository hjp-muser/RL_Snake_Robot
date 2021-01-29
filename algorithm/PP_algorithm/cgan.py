import tensorflow as tf
import numpy as np
from matplotlib import gridspec
from data_operation import read_csv
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
import json


def generate_noise(m, n):
    """
    :param m: batch
    :param n: dimension
    :return: input noise
    """
    return np.random.uniform(-1, 1, size=[m, n])


def plot_samples(samples):
    fig = plt.figure()
    # gs = gridspec.GridSpec(5, 1)
    # gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        # axes = plt.subplot(gs[i])
        # plt.axis("off")
        # axes.set_xlim(0, 505)
        # axes.set_ylim(-0.6, 0.6)
        filter_result = np.convolve(sample, np.ones(10,)/10, mode='same')
        # fft_result = np.fft.fft(sample)
        # fft_result = np.where(np.absolute((fft_result) < np.mean(np.absolute(fft_result))), 0, fft_result)
        # ifft_result = np.real(np.fft.ifft(fft_result))
        plt.plot(filter_result)
    return fig

# def plot_samples(samples):
#     fig = plt.figure(figsize=(5, 5))
#     gs = gridspec.GridSpec(5, 5)
#     gs.update(wspace=0.02, hspace=0.02)
#
#     for i, sample in enumerate(samples):
#         axes = plt.subplot(gs[i])
#         plt.axis("off")
#         axes.set_xticklabels([])  # set x label font
#         axes.set_yticklabels([])
#         axes.set_aspect('equal')  # length of x,y is equal
#         plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#
#     return fig


class CGAN:

    # @staticmethod
    # def _generator(z, y, vars):
    #     G_w1, G_w2, G_w3, G_b1, G_b2, G_b3 = vars
    #     G_inputs = tf.concat(values=[z, y], axis=1)
    #     G_z1 = tf.matmul(G_inputs, G_w1) + G_b1
    #     G_h1 = tf.nn.relu(G_z1)
    #     G_z2 = tf.matmul(G_h1, G_w2) + G_b2
    #     G_h2 = tf.nn.relu(G_z2)
    #     G_logits = tf.matmul(G_h2, G_w3) + G_b3
    #     G_prob = tf.nn.sigmoid(G_logits)
    #     return G_prob

    # @staticmethod
    # def _discriminator(x, y, vars):
    #     D_w1, D_w2, D_w3, D_b1, D_b2, D_b3 = vars
    #     D_inputs = tf.concat(values=[x, y], axis=1)
    #     D_z1 = tf.matmul(D_inputs, D_w1) + D_b1
    #     D_h1 = tf.nn.leaky_relu(D_z1)
    #     D_z2 = tf.matmul(D_h1, D_w2) + D_b2
    #     D_h2 = tf.nn.leaky_relu(D_z2)
    #     D_logits = tf.matmul(D_h2, D_w3) + D_b3
    #     D_labels = tf.rint(tf.nn.sigmoid(D_logits))
    #     return D_labels, D_logits

    @staticmethod
    def _discriminator(x, y, vars):
        D_w1, D_w2, D_w3, D_b1, D_b2, D_b3 = vars
        D_inputs = tf.concat(values=[x, y], axis=1)
        D_z1 = tf.matmul(D_inputs, D_w1) + D_b1
        D_z1 = tf.concat(values=[D_z1, y], axis=1)
        D_h1 = tf.nn.leaky_relu(D_z1)
        D_z2 = tf.matmul(D_h1, D_w2) + D_b2
        D_z2 = tf.concat(values=[D_z2, y], axis=1)
        D_h2 = tf.nn.leaky_relu(D_z2)
        D_logits = tf.matmul(D_h2, D_w3) + D_b3
        D_labels = tf.rint(tf.nn.sigmoid(D_logits))
        return D_labels, D_logits

    # @staticmethod
    # def _generator(z, y, vars):
    #     G_w1, G_w2, G_b1, G_b2 = vars
    #     G_inputs = tf.concat(values=[z, y], axis=1)
    #     G_z1 = tf.matmul(G_inputs, G_w1) + G_b1
    #     G_h1 = tf.nn.leaky_relu(G_z1)
    #     G_logits = tf.matmul(G_h1, G_w2) + G_b2
    #     G_prob = tf.nn.sigmoid(G_logits)
    #     return G_logits

    @staticmethod
    def _generator(z, y, vars):
        G_w1, G_w2, G_b1, G_b2 = vars
        G_inputs = tf.concat(values=[z, y], axis=1)
        G_z1 = tf.matmul(G_inputs, G_w1) + G_b1
        G_z1 = tf.concat(values=[G_z1, y], axis=1)
        G_h1 = tf.nn.leaky_relu(G_z1)
        G_logits = tf.matmul(G_h1, G_w2) + G_b2
        G_prob = tf.nn.sigmoid(G_logits)
        return G_logits

    def __init__(self, data_dim, label_dim, noise_dim, learning_rate=0.001, batch_size=64, log_interval=10):
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.noise_dim = noise_dim
        self.lr = learning_rate
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.hidden_size_1 = 64
        self.hidden_size_2 = 64
        self.hidden_size_3 = 64
        self.graph = tf.Graph()
        self.model_path = 'model/cgan_model.ckpt'
        self.mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

        with self.graph.as_default():
            self.data_ph = tf.placeholder(tf.float32, shape=(None, self.data_dim))
            self.label_ph = tf.placeholder(tf.float32, shape=(None, self.label_dim))
            self.noise_ph = tf.placeholder(tf.float32, shape=(None, self.noise_dim))

            G_input_dim = self.noise_dim + self.label_dim
            G_w1 = tf.Variable(tf.random_normal(shape=[G_input_dim, self.hidden_size_1],
                                                stddev=1. / tf.sqrt(G_input_dim / 2.0)))
            G_b1 = tf.Variable(tf.zeros(shape=[self.hidden_size_1]))
            G_w2 = tf.Variable(tf.random_normal(shape=[self.hidden_size_1+self.label_dim, self.data_dim],
                                                stddev=1. / tf.sqrt(self.hidden_size_1 / 2.0)))
            G_b2 = tf.Variable(tf.zeros(shape=[self.data_dim]))
            self.vars_G = [G_w1, G_w2, G_b1, G_b2]
            # G_w2 = tf.Variable(tf.random_normal(shape=[self.hidden_size_1, self.hidden_size_2],
            #                                     stddev=1. / tf.sqrt(self.hidden_size_1 / 2.0)))
            # G_b2 = tf.Variable(tf.zeros(shape=[self.hidden_size_2]))
            # G_w3 = tf.Variable(tf.random_normal(shape=[self.hidden_size_2, self.data_dim],
            #                                     stddev=1. / tf.sqrt(self.hidden_size_2 / 2.0)))
            # G_b3 = tf.Variable(tf.zeros(shape=[self.data_dim]))
            # self.vars_G = [G_w1, G_w2, G_w3, G_b1, G_b2, G_b3]

            # The Discrimination Net model
            D_input_dim = self.data_dim + self.label_dim
            D_w1 = tf.Variable(tf.random_normal(shape=[D_input_dim, self.hidden_size_1],
                                                stddev=1. / tf.sqrt(D_input_dim / 2.0)))
            D_b1 = tf.Variable(tf.zeros(shape=[self.hidden_size_1]))
            # D_w2 = tf.Variable(tf.random_normal(shape=[self.hidden_size_1, 1],
            #                                     stddev=1. / tf.sqrt(self.hidden_size_1 / 2.0)))
            # D_b2 = tf.Variable(tf.zeros(shape=[1]))
            # self.vars_D = [D_w1, D_w2, D_b1, D_b2]
            D_w2 = tf.Variable(tf.random_normal(shape=[self.hidden_size_1+self.label_dim, self.hidden_size_2],
                                                stddev=1. / tf.sqrt(self.hidden_size_2 / 2.0)))
            D_b2 = tf.Variable(tf.zeros(shape=[self.hidden_size_2]))
            D_w3 = tf.Variable(tf.random_normal(shape=[self.hidden_size_2+self.label_dim, 1],
                                                stddev=1. / tf.sqrt(self.hidden_size_2 / 2.0)))
            D_b3 = tf.Variable(tf.zeros(shape=[1]))
            self.vars_D = [D_w1, D_w2, D_w3, D_b1, D_b2, D_b3]

            self.G_samples = self._generator(self.noise_ph, self.label_ph, self.vars_G)
            D_real_labels, D_logits = self._discriminator(self.data_ph, self.label_ph, self.vars_D)
            D_fake_labels, D_log_fake = self._discriminator(self.G_samples, self.label_ph, self.vars_D)

            D_regularization = tf.nn.l2_loss(D_w1) + tf.nn.l2_loss(D_w2) + tf.nn.l2_loss(D_w3) + \
                                tf.nn.l2_loss(D_b1) + tf.nn.l2_loss(D_b2) + tf.nn.l2_loss(D_b3)
            G_regulariazation = tf.nn.l2_loss(G_w1) + tf.nn.l2_loss(G_w2) + \
                                tf.nn.l2_loss(G_b1) + tf.nn.l2_loss(G_b2)

            D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
            D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_log_fake, labels=tf.zeros_like(D_log_fake)))

            with tf.name_scope('d_loss'):
                self.D_loss = D_loss_real + D_loss_fake
                tf.summary.scalar('d_loss', self.D_loss)
            with tf.name_scope('g_loss'):
                bias = self.G_samples - tf.reduce_mean(self.G_samples, axis=1, keepdims=True)
                norm = tf.reduce_sum(bias * bias, axis=1)
                acorloss = tf.reduce_mean(tf.reduce_sum(bias[:, :-1] * bias[:, 1:], axis=1) / norm)
                self.G_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_log_fake, labels=tf.ones_like(D_log_fake))) - 10 * acorloss + 0.001 * G_regulariazation
                tf.summary.scalar('g_loss', self.G_loss)

            # global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(self.lr, global_step=1000000000, decay_steps=10000000, decay_rate=0.99)
            self.D_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.D_loss, var_list=self.vars_D)
            self.G_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.G_loss, var_list=self.vars_G)

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(r'./log2', tf.get_default_graph())
            self.saver = tf.train.Saver()

    def learning(self, data=None, labels=None, num_steps=100000):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            if not os.path.exists('out2/'):
                os.makedirs('out2/')
            i = 0
            for n in range(num_steps):
                if n % self.log_interval == 0:
                    n_samples = 1
                    Z_sample = generate_noise(n_samples, self.noise_dim)
                    # y_samples = np.zeros(shape=[n_samples, self.label_dim])
                    # # y_samples[:, 6] = 1
                    # y_samples[:, i%16] = 1
                    y_samples = np.array(label_type[i%4])
                    y_samples = y_samples[np.newaxis, :]

                    samples = sess.run(self.G_samples, feed_dict={self.noise_ph: Z_sample, self.label_ph: y_samples})

                    fig = plot_samples(samples)
                    plt.savefig('out2/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    i += 1
                    plt.close(fig)

                shuffle_ix = np.random.permutation(np.arange(data.shape[0]))
                for begin in range(0, data.shape[0], self.batch_size):
                    end = min(begin + self.batch_size, data.shape[0])
                    train_data = data[shuffle_ix[begin:end], :]
                    train_labels = labels[shuffle_ix[begin:end], :]
                    Z_sample = generate_noise(end-begin, self.noise_dim)
                # Z_sample = generate_noise(self.batch_size, self.noise_dim)
                # train_data, train_labels = self.mnist_dataset.train.next_batch(self.batch_size)
                    _, D_loss_curr = sess.run([self.D_optimizer, self.D_loss],
                                              feed_dict={self.data_ph: train_data, self.noise_ph: Z_sample, self.label_ph: train_labels})
                    _, G_loss_curr = sess.run([self.G_optimizer, self.G_loss],
                                              feed_dict={self.noise_ph: Z_sample, self.label_ph: train_labels})
                    result = sess.run(self.merged,
                                      feed_dict={self.data_ph: train_data, self.noise_ph: Z_sample, self.label_ph: train_labels})
                    self.writer.add_summary(result, n)

                if n % self.log_interval == 0:
                    print('Iter: {}'.format(n))
                    print('D loss: ', D_loss_curr)
                    print('G_loss: ', G_loss_curr)
                    print()


if __name__ == '__main__':
    batch = 300
    X_dim = 500
    y_dim = 16
    Z_dim = 500
    lr = 0.001
    inputs = read_csv('data/1.csv', colums='noise')
    inputs = np.array([json.loads(x) for x in inputs])
    labels = read_csv('data/1.csv', colums='gray')
    gray_labels = np.array([np.array(list(x)).astype(int) for x in labels])
    onehot_labels = []
    for label in gray_labels:
        label_num = 8 * label[0] + 4 * label[1] + 2 * label[2] + label[3]
        onehot_label = [0] * 16
        onehot_label[label_num] = 1
        onehot_labels.append(onehot_label)
    onehot_labels = np.array(onehot_labels)
    label_type = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    cgan = CGAN(X_dim, y_dim, Z_dim, lr, batch)
    cgan.learning(inputs, onehot_labels)

    # sin function test
    # batch = 64
    # X_dim = 500
    # y_dim = 16
    # Z_dim = 100
    # lr = 0.0001
    # a = np.sin(1 * np.linspace(0, 10, 500))
    # b = np.cos(2 * np.linspace(0, 10, 500))
    # c = np.sin(3 * np.linspace(0, 10, 500))
    # d = np.cos(4 * np.linspace(0, 10, 500))
    # e = np.sin(5 * np.linspace(0, 10, 500))
    # f = np.cos(6 * np.linspace(0, 10, 500))
    # g = np.sin(7 * np.linspace(0, 10, 500))
    # o = np.cos(8 * np.linspace(0, 10, 500))
    # p = np.sin(9 * np.linspace(0, 10, 500))
    # q = np.cos(10 * np.linspace(0, 10, 500))
    # r = np.sin(11 * np.linspace(0, 10, 500))
    # s = np.cos(12 * np.linspace(0, 10, 500))
    # t = np.sin(13 * np.linspace(0, 10, 500))
    # u = np.cos(14 * np.linspace(0, 10, 500))
    # v = np.sin(15 * np.linspace(0, 10, 500))
    # w = np.cos(16 * np.linspace(0, 10, 500))
    #
    # input_type = [a, b, c, d, e, f, g, o, p, q, r, s, t, u, v, w]
    # # label_type = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    # label_type = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    # inputs = []
    # labels = []
    # rand_id = np.random.randint(0, 16, 6400)
    # for i in rand_id:
    #     inputs.append(input_type[i] + 0.1 * np.random.rand(500))  # + 0.1 * np.random.rand(500)
    #     labels.append(label_type[i])
    # inputs = np.array(inputs)
    # labels = np.array(labels)
    # cgan = CGAN(X_dim, y_dim, Z_dim, lr, batch)
    # cgan.learning(inputs, labels)


    # sin function test2
    # batch = 64
    # X_dim = 500
    # y_dim = 4
    # Z_dim = 100
    # lr = 0.0001
    # a = 1 * np.linspace(0, 10, 500)
    # b = 2 * np.linspace(0, 10, 500)
    # c = 3 * np.linspace(0, 10, 500)
    # d = 4 * np.linspace(0, 10, 500)
    #
    # input_type = [a, b, c, d]
    # label_type = [[0, 0, 0, 1], [0, 0, 1, 0],
    #               [0, 1, 0, 0], [1, 0, 0, 0]]
    # inputs = []
    # labels = []
    # rand_id = np.random.randint(0, 4, 6400)
    # j = 0
    # plt.figure()
    # for i in rand_id:
    #     inputs.append(np.sin(input_type[i] + 0.2 * np.random.rand(500)) + 0.2 * np.random.rand(500))  # + 0.1 * np.random.rand(500)
    #     labels.append(label_type[i])
    # inputs = np.array(inputs)
    # labels = np.array(labels)
    # cgan = CGAN(X_dim, y_dim, Z_dim, lr, batch)
    # cgan.learning(inputs, labels)


