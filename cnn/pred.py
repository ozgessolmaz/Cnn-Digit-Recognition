import tensorflow as tf

model = tf.keras.models.load_model("odev_model.keras")

model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
