import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


predictions = model(x_train[:1]).numpy() # this model returns a vector of logits or "log-odds"
prob_predictions = tf.nn.softmax(predictions).numpy() # converts logs to probibility
# print(prob_predictions)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# not sure why but this only gives a correct answer when I use prob_predictions
print(loss_fn(y_train[:1], prob_predictions).numpy())
# print(loss_fn(y_train[:1], predictions).numpy())

model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5, verbose=2)
model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))