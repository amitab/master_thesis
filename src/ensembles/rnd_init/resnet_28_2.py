import keras
import tensorflow as tf
import tensorflow_addons as tfa
import keras_wrn

import sys

from keras.optimizers import SGD

prefix = sys.argv[1]

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

shape, classes = (32, 32, 3), 10

model_depth = 28
model_width = 2

model = keras_wrn.build_model(shape, classes, model_depth, model_width)
# optimizer = tfa.optimizers.SGDW(weight_decay=0.0005, momentum=0.9, learning_rate=0.1)
# model.compile(optimizer, "categorical_crossentropy", ["accuracy"])
opt = SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


def scheduler(epoch, lr):
  if epoch in (80, 100, 120):
    lr = lr * 0.2 
  return lr

def model_save_callback(epoch, logs):
    if epoch % 50 == 0 and epoch != 0:
        model.save('model_ckpt/{}_resnet_28_2_{}'.format(prefix, epoch))

save_call = tf.keras.callbacks.LambdaCallback(on_epoch_end=model_save_callback)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(x_train, y_train, batch_size=125, epochs=140, callbacks=[callback, save_call])

model.save('models/{}_resnet_{}_{}'.format(prefix, model_depth, model_width))

results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)
