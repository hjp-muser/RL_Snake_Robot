import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import math
from data_operation import read_csv
import json


class EncodeNet:
    def __init__(self, input_dim: int, output_dim: int, batch_size=128, rc=0.0002, lr=0.0099, log_inteval=10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rc = rc  # regularization coefficient
        self.lr = lr
        self.batch_size = batch_size
        self.log_interval = log_inteval
        self.graph = tf.Graph()
        self.kernel_num = 8
        self.hidden_size_1 = 64
        self.hidden_size_2 = 64
        self.hidden_size_3 = 64
        self.model_path = "model.ckpt"

        with self.graph.as_default():
            self.input_ph = tf.placeholder(tf.float32, shape=(None, input_dim))
            self.label_ph = tf.placeholder(tf.float32, shape=(None, output_dim))

            # 核层
            # 定义径向基层
            guassian_kernel_center = tf.Variable(tf.random_normal([self.kernel_num, self.input_dim]))
            guassian_kernel_std = tf.Variable(tf.random_normal([self.kernel_num]))

            # 隐藏层1
            # hidden_layer_weights_1 = tf.Variable(tf.random.truncated_normal([self.input_dim, self.hidden_size_1],
            #                                                                 stddev=math.sqrt(2.0 / self.batch_size)))
            hidden_layer_weights_1 = tf.Variable(tf.random.truncated_normal([self.kernel_num, self.hidden_size_1],
                                                                            stddev=math.sqrt(2.0 / self.kernel_num)))
            hidden_layer_bias_1 = tf.Variable(tf.zeros([self.hidden_size_1]))

            # 隐藏层2
            hidden_layer_weights_2 = tf.Variable(tf.random.truncated_normal([self.hidden_size_1, self.hidden_size_2],
                                                                            stddev=math.sqrt(2.0 / self.hidden_size_1)))
            hidden_layer_bias_2 = tf.Variable(tf.zeros([self.hidden_size_2]))

            # 隐藏层3
            hidden_layer_weights_3 = tf.Variable(tf.random.truncated_normal([self.hidden_size_2, self.hidden_size_3],
                                                                            stddev=math.sqrt(2.0 / self.hidden_size_2)))
            hidden_layer_bias_3 = tf.Variable(tf.zeros([self.hidden_size_3]))

            # 输出层
            # out_weights = tf.Variable(tf.random.truncated_normal([self.hidden_size_2, self.output_dim],
            #                                                      stddev=math.sqrt(2.0 / self.hidden_size_2)))
            out_weights = tf.Variable(tf.random.truncated_normal([self.hidden_size_3, self.output_dim],
                                                                 stddev=math.sqrt(2.0 / self.hidden_size_3)))
            out_bias = tf.Variable(tf.zeros([self.output_dim]))

            # self.weights = [hidden_layer_weights_1, hidden_layer_bias_1, hidden_layer_weights_2, hidden_layer_bias_2,
            #                 out_weights, out_bias]
            self.weights = [hidden_layer_weights_1, hidden_layer_bias_1, hidden_layer_weights_2, hidden_layer_bias_2,
                            hidden_layer_weights_3, hidden_layer_bias_3, out_weights, out_bias]

            z0 = self.kernel(self.input_ph, guassian_kernel_center, guassian_kernel_std)

            # z1 = tf.matmul(self.input_ph, hidden_layer_weights_1) + hidden_layer_bias_1
            z1 = tf.matmul(z0, hidden_layer_weights_1) + hidden_layer_bias_1
            h1 = tf.nn.relu(z1)

            z2 = tf.matmul(h1, hidden_layer_weights_2) + hidden_layer_bias_2
            h2 = tf.nn.relu(z2)

            z3 = tf.matmul(h2, hidden_layer_weights_3) + hidden_layer_bias_3
            h3 = tf.nn.relu(z3)
            # self.logits = tf.matmul(h2, out_weights) + out_bias
            self.logits = tf.matmul(h3, out_weights) + out_bias

            # L2正则化
            # regularization = tf.nn.l2_loss(hidden_layer_weights_1) + tf.nn.l2_loss(hidden_layer_weights_2) \
            #                  + tf.nn.l2_loss(out_weights)
            regularization = tf.nn.l2_loss(hidden_layer_weights_1) + tf.nn.l2_loss(hidden_layer_weights_2) + \
                             tf.nn.l2_loss(hidden_layer_weights_3) + tf.nn.l2_loss(out_weights)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.label_ph, logits=self.logits) + self.rc * regularization)

            # self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            self.predictions = tf.rint(tf.nn.sigmoid(self.logits))

            self.saver = tf.train.Saver()

    # 高斯核函数(c为中心，s为标准差)
    # 高斯核函数(c为中心，s为标准差)
    def kernel(self, x, c, s):  # 训练时使用
        x1 = tf.tile(x, [1, self.kernel_num])  # 将x水平复制 hidden次
        x2 = tf.reshape(x1, [-1, self.kernel_num, self.input_dim])
        dist = tf.reduce_sum(tf.pow((x2 - c), 2), 2)
        return tf.exp(-dist / (2 * tf.pow(s, 2)))

    def learning(self, inputs, labels, num_steps=10000):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            for step in range(num_steps):
                for begin in range(0, inputs.shape[0], self.batch_size):
                    end = min(begin + self.batch_size, inputs.shape[0])
                    _, loss, predictions = session.run([self.optimizer, self.loss, self.predictions],
                                                       feed_dict={self.input_ph: inputs[begin:end, :],
                                                                  self.label_ph: labels[begin:end, :]})
                if step % 1000 == 0:
                    print('Loss at step %d: %f' % (step, loss))
                    predictions = session.run(self.predictions, feed_dict={self.input_ph: inputs})
                    print(predictions[0])
                    print('Training accuracy: %.1f%%' % self.accuracy(predictions, labels))
                    self.saver.save(session, self.model_path)

    def accuracy(self, predictions, labels):
        return 100.0 * np.sum(np.all(predictions == labels, axis=1)) / predictions.shape[0]

    def predict(self, inputs):
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, self.model_path)
            predictions = session.run(self.predictions, feed_dict={self.input_ph: inputs})
        return predictions

    def get_weights(self):
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, self.model_path)
            weights = session.run(self.weights)
        return weights

    def cal_logits(self, inputs):
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, self.model_path)
            logits = session.run(self.logits, feed_dict={self.input_ph: inputs})
        return logits


if __name__ == "__main__":
    inputs = read_csv('data/1.csv', colums='prior')
    inputs = np.array([json.loads(x) for x in inputs])
    labels = read_csv('data/1.csv', colums='gray')
    labels = np.array([np.array(list(x)).astype(int) for x in labels])

    encode_net = EncodeNet(inputs.shape[1], labels.shape[1], batch_size=256)
    encode_net.learning(inputs=inputs, labels=labels, num_steps=20000)
    weights = encode_net.get_weights()
    print(weights)
