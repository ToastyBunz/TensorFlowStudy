import os
import tensorflow as tf
from tensorflow import keras
# print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Original model

def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

model = create_model()
print(model.summary())

# Save checkpoints from trained model
# you can pick it up where it left off or use a totally trained model
# this method can continually save before, during and after training

checkpoint_path = 'trianing_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# #create a callback that saves model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# #Train model with new callback
# model.fit(train_images,
#           train_labels,
#           epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

print(os.listdir(checkpoint_dir))


# as long as two models share the same architecture you can share weights between them.
# to restore weights from one model to another, create a model of the same architecture

# Create basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Untrained Model, accuracy: {:5.2f}%'.format(100*acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accurace: {:5.2f}f%'.format(100*acc))


# Checkpoint call backs
# Train new model and save uniquely named checkpoints every five epochs

# Include the epoch in the file name (uses 'str.format')
checkpoint_path = 'training_t/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Created a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size
)

# Create a new model instance
model = create_model()

# Save the weights using the 'checkpoint_path' format
model.save_weights(checkpoint_path.format(epoch=0))

#Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=50,
          batch_size=batch_size,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

print(os.listdir(checkpoint_dir))
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# to test, reset the model and load the latest checkpoint

# Create a new model instance
model = create_model()

# load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model 2, accuracy: {:5.2f}%'.format(100 * acc))




# TODO fix this block
# manually saved weights: use tf,keras.Model.save_weights

# hand_checkpoint_path = './checkpoints/my_checkpoint'
# hand_checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # Save the weights
# model.save_weights(hand_checkpoint_path)
#
# #create a new model instance
# model = create_model()
#
# # restore weights
# model.load_weights(hand_checkpoint_path)
#
# # evaluate model
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, Hand Saved, Accuracy: {:5.2f}%".format(100 * acc))

# TODO figure out why this block is not working
# # Save entire model
#
# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
#
# # save entire model
# model.save_model('saved_model/my_model')
#
# #my model_directory
# ls saved_model
# #Contains an asset forlder, saved_model.pb and vaiables folder
# ls saved_model/my_model
#
# # reaload a fresh keras model from the saved model
#
# new_model = tf.keras.models.load_model('saved_model/my_model')
#
# # Check its architecture
# new_model.summary
#
# # evaluate restored model
# loss, acc - new_model.evaluate(test_images, test_labels, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
#
# print(new_model.predict(test_images).shape)

# HDF5
model = create_model()
model.fit(train_images, train_labels, epochs = 5)

# save the entire model to a HDF5 file .h5

model.save('my_model.h5')

new_model = tf.keras.models.load_model('my_model.h5')
print(new_model.summary())
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model H5, accuracy: {:5.2f}%'.format(100*acc))


