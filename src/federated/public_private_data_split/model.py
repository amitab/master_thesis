from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import tensorflow as tf

import numpy as np

tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

#from dp_optimizer_v2 import DPGradientDescentGaussianOptimizer


train, test = tf.keras.datasets.mnist.load_data()
train_data, train_labels = train
test_data, test_labels = test

train_data = np.array(train_data, dtype=np.float32) / 255
test_data = np.array(test_data, dtype=np.float32) / 255

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

train_labels = np.array(train_labels, dtype=np.int32)
test_labels = np.array(test_labels, dtype=np.int32)

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

assert train_data.min() == 0.
assert train_data.max() == 1.
assert test_data.min() == 0.
assert test_data.max() == 1.


shared_percent = 50 / 100
shared_train_data_size = int(len(train_data) * shared_percent)
num_models = 2

shared_train_data = train_data[:shared_train_data_size]
shared_train_labels = train_labels[:shared_train_data_size]
private_train_data = np.split(train_data[shared_train_data_size:], num_models)
private_train_labels = np.split(
    train_labels[shared_train_data_size:], num_models)


epochs = 10
batch_size = 250

l2_norm_clip = 1.5
noise_multiplier = 0
num_microbatches = batch_size
learning_rate = 0.25

if batch_size % num_microbatches != 0:
    raise ValueError(
        'Batch size should be an integer multiple of the number of microbatches')


main_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 8,
                            strides=2,
                            padding='same',
                            activation='relu',
                            input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(32, 4,
                           strides=2,
                           padding='valid',
                           activation='relu'),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# old_weights = main_model.get_weights()

main_model.summary()
og_file_name = 'models/og_{}_{}.h5'.format(int(shared_percent * 100), num_models)
main_model.save(og_file_name)

for i in range(num_models):
    print("Training model - {}".format(i))
    
    model_train_data = np.vstack((shared_train_data, private_train_data[i]))
    model_train_labels = np.vstack(
        (shared_train_labels, private_train_labels[i]))

    model = tf.keras.models.load_model(og_file_name)
    # Will compile mess up optimizer state?
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model.set_weights(old_weights)

    model.fit(model_train_data, model_train_labels,
              epochs=epochs,
              validation_data=(test_data, test_labels),
              batch_size=batch_size)

    model_name = "models/clip_1_5_NO_{}_{}_{}.h5".format(i, int(shared_percent * 100), num_models)
    model.save(model_name)
