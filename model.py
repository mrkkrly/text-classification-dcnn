import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, Embedding, Dropout, Dense

class DCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedd_dim, n_classes,
                 n_filters, maxlen, n_units, dropout, training=False,
                 name="dcnn"):

        super(DCNN, self).__init__(name=name)
        self.embedding = Embedding(vocab_size, embedd_dim)
        self.dropout = Dropout(dropout)
        self.conv_1 = Conv1D(n_filters, kernel_size=2, padding="valid", activation="relu")
        self.pool_1 = GlobalMaxPool1D()
        self.conv_2 = Conv1D(n_filters, kernel_size=3, padding="valid", activation="relu")
        self.pool_2 = GlobalMaxPool1D()
        self.conv_3 = Conv1D(n_filters, kernel_size=4, padding="valid", activation="relu")
        self.pool_3 = GlobalMaxPool1D()
        self.dense_1 = Dense(n_units, activation="relu")
        if n_classes == 2:
            self.outputs_layer = Dense(1, activation="sigmoid")
        else:
            self.outputs_layer = Dense(n_classes, activation="softmax")

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.conv_1(x)
        x_1 = self.pool_1(x_1)
        x_2 = self.conv_2(x)
        x_2 = self.pool_2(x_2)
        x_3 = self.conv_3(x)
        x_3 = self.pool_3(x_3)
        merged = tf.concat([x_1, x_2, x_3], axis=-1)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        outputs = self.outputs_layer(merged)
        return outputs
