import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import sys

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # self.conv1 = Conv2D(32, 3, activation='relu')
    # self.flatten = Flatten()
    self.flatten = Flatten( input_shape=(None,2,14,2,10,10))
    self.d0 = Dense(1000, activation='relu',)
    self.d1 = Dense(2, activation='softmax')

  def call(self, x):
    # x = self.conv1(x)
    # x = self.flatten(x)
    x = self.flatten(x)

    x = self.d0(x)
    x = self.d1(x)
    # x = self.d2(x)

    return x

# Create an instance of the model




def main():
    args = sys.argv
    data_file_name = args[1]
    out_name = args[2]
    test_len = 1000
    data = np.load(data_file_name, allow_pickle=True)
    data_t_bool = np.load(data_file_name.replace('.npy','_t.npy'), allow_pickle=True)
    data_t = np.zeros([len(data_t_bool),2])
    data_t[data_t_bool.astype(bool)][:,0] = 1
    data_t[~data_t_bool.astype(bool)][:,1] = 1

    mask_ind = np.random.choice(len(data), test_len)
    test_mask = np.zeros(len(data),dtype=bool)
    test_mask[mask_ind] = True

    test_data = data[test_mask]
    test_data_t = data_t[test_mask]
    data_t = data_t[~test_mask]
    data = data[~test_mask]
    x_train = data
    y_train = data_t
    # y_train_human = data[:, 162:164]
    x_test = test_data
    y_test = test_data_t
    batch_size = 16

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    model = MyModel()


    # loss_object = tf.keras.losses.SquaredHinge()
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # loss_object = tf.keras.losses.BinaryCrossentropy(reduction='sum_over_batch_size')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


    @tf.function
    def train_step(images, labels):

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            # labels=labels.reshape(32,2)
            print(predictions)
            loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # print(loss)
            train_loss(loss)
            train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)


    EPOCHS = 10
    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            # labels = labels.reshape(32,2)
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}, '
                f'Test Loss: {test_loss.result()}, '
                f'Test Accuracy: {test_accuracy.result() * 100}'
        )

    tf.keras.models.save_model(model,"tf_model")
    raise ValueError
if __name__ == "__main__":
    main()
