import keras
import keras_wrn

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

shape, classes = (32, 32, 3), 10

model_depth = 16
model_width = 4

model = keras_wrn.build_model(shape, classes, model_depth, model_width)
model.compile("adam", "categorical_crossentropy", ["accuracy"])

model.fit(x_train, y_train, epochs=10)

model.save('models/resnet_{}_{}'.format(model_depth, model_width))

results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)