import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, Sequential

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
batchsz = 128
# 批量大小
# the most frequents words
total_words = 10000
max_review_len = 80
embedding_len = 100
# 词向量特征长度f
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# x_train:[b, 80]
# x_test: [b, 80]
# 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
# 构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batchsz, units])]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        # [b, 80, 100], h_dim:64
        self.run_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True): train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # [b, 80, 100] => [b, 64]
        state0 = self.state0
        for word in tf.unstack(x, axis=1):
            # word: [b, 100]
            out, state1 = self.run_cell0(word, state0)
            state0 = state1
        # out [b, 64] => [b, 1]
        x = self.outlayer(out)
        prob = tf.sigmoid(x)
        return prob


def main():
    units = 64
    epochs = 4
    model = MyRNN(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)


if __name__ == '__main__':
    main()