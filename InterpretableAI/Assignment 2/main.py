import numpy as np
import tensorflow as tf

from keras import models, layers

tf.config.list_physical_devices("GPU")

# # Fixing mnist data
# (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = (
#     tf.keras.datasets.mnist.load_data(path="mnist.npz")
# )

# mnist_x_train, mnist_x_test = (
#     mnist_x_train.astype("float32") / 255.0,
#     mnist_x_test.astype("float32") / 255.0,
# )
# # CNN for mnist
# model = models.Sequential()
# # Layer 1 Conv2d, 32 filters, 3x3
# model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
# model.add(layers.BatchNormalization())
# # Layer 2 max pool 2x2
# model.add(layers.MaxPooling2D(2, 2))
# # Layer 3 conv2d 64, 3x3
# model.add(layers.Conv2D(64, (3, 3), activation="relu"))
# model.add(layers.BatchNormalization())
# # Layer 4
# model.add(layers.MaxPooling2D(2, 2))
# # Layer 5
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation="relu"))
# model.add(layers.Dropout(0.35))
# # Layer 6
# model.add(layers.Dense(10, activation="softmax"))

# model.summary()


# model.compile(
#     optimizer="adam",
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=["accuracy"],
# )

# history = model.fit(
#     mnist_x_train,
#     mnist_y_train,
#     epochs=10,
#     validation_data=(mnist_x_test, mnist_y_test),
# )

# # Fixing cifar10 data
(cifar_image_train, cifar_label_train), (cifar_image_test, cifar_label_test) = (
    tf.keras.datasets.cifar10.load_data()
)

cifar_image_train = cifar_image_train.astype("float32")
cifar_image_test = cifar_image_test.astype("float32")

mean_training = np.mean(cifar_image_train)
std_dev_training = np.std(cifar_image_train)

cifar_image_train = (cifar_image_train - mean_training) / std_dev_training
cifar_image_test = (cifar_image_test - mean_training) / std_dev_training

model_cifar = models.Sequential()
# Layer 1
model_cifar.add(
    layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"
    )
)

layers.Conv2D(
    64, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
)
model_cifar.add(layers.BatchNormalization())

# Layer 2
model_cifar.add(layers.MaxPooling2D(2, 2))

# Layer 3
model_cifar.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model_cifar.add(layers.BatchNormalization())

# Layer 4
model_cifar.add(layers.MaxPooling2D(2, 2))

# Layer 5
model_cifar.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
model_cifar.add(layers.BatchNormalization())

# Layer 6
model_cifar.add(layers.MaxPooling2D(2, 2))

# Layer 7
model_cifar.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
model_cifar.add(layers.BatchNormalization())

# Layer 8
model_cifar.add(layers.MaxPooling2D(2, 2))

model_cifar.add(layers.Flatten())
model_cifar.add(layers.Dense(128, activation="relu"))
model_cifar.add(layers.Dropout(0.35))
model_cifar.add(layers.Dense(10, activation="softmax"))


model_cifar.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

history = model_cifar.fit(
    cifar_image_train,
    cifar_label_train,
    epochs=10,
    validation_data=(cifar_image_test, cifar_label_test),
)

model_cifar.save("cifar_cnn_model.keras")
