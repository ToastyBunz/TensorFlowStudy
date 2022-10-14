# https://www.tensorflow.org/guide/keras/save_and_serialize

'''This is a collection of examples NOT MEANT TO BE RUN ALL AT ONCE'''

# # Quickstart
#
# # Saving Model
# model = ... # Get model (Sequential, Functional Model, or model subclass)
# model.save('path/to/location')
#
# #Loading Model back
# from tensorflow import keras
# model = keras.models.load_model('path/to/location')

### Now full thing

import numpy as np
import tensorflow as tf
from tensorflow import keras

## whole model saving
# architecture, weights, compile, optimizer and its state



# SavedModel format

def get_model():
    # create a simple model
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
model = get_model()

# Train the model
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))

model.fit(test_input, test_target)

# calling 'save('my_model')' creates a SavedModel folder ('my_model')
model.save('kitkat_model')

# reconstruct model
reconstructed_model = keras.models.load_model('kitkat_model')

# lets check
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

reconstructed_model.fit(test_input, test_target)


### Custom objects

class CustomModel(keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {'hidden_units': self.hidden_units}

    @classmethod
    def form_config(cls, config):
        return cls(**config)

model = CustomModel([16, 16, 10])

#build the model by calling it
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model.save('kitkat_model')

# Option 1: Load with the custom_object argument.
loaded_1 = keras.models.load_model(
    'kitkat_model', custom_objects = {'CustomModel': CustomModel}
)

# Option 2: Load without CustomModel class
# Delete the custom-defined model class to ensure that the loader does not have access to it
del CustomModel
loaded_2 = keras.models.load_model('kitkat_model')
np.testing.assert_allclose(loaded_1(input_arr), outputs)
np.testing.assert_allclose(loaded_2(input_arr), outputs)

print('Origonal model:', model)
print('Model loaded with custom objects:', loaded_1)
print('Model loaded without the custom object class:', loaded_2)

# Model 1 uses config and CustomModel class.
# Model 2 loaded dynamically creating model class that acts like the origonal model


# New to TF 2.4
# argument save_traces has been added to model.save, allows toggle function tracing
# when save_traces=False all custom objs must define get_config/from_config
# When loading, custom objs must be passed to the custom_objects argument.
# save_traces=False rudecs the disk space used by the SavedModel and saving time

## Saving H5 format

# model = get_model()
#
# # Train teh model
# test_input = np.random.random((128, 32))
# test_target = np.random.random((128, 1))
# model.fit(test_input, test_target)
#
# # Calling 'save('kitkat.h5')' creates a h5 file
# model.save('kitkat_model.h5')
#
# # it can be sued to reconstruct the model idenitaclly
# reconstructed_model = keras.models.load_model('kitkat_model.h5')
#
# # check
# np.testing.assert_allclose(
#     model.predict(test_input), reconstructed_model.predict(test_input)
# )
#
# # the reconstructed model already compiled and has retained the optimizer state
# # so you can continue training
# reconstructed_model.fit(test_input, test_target)

# Defining config methods

class CustomLayers(keras.layers.Layer):
    def __init__(self, a):
        self.var = tf.Variable(a, name='var_a')

    def call(self, inputs, training=False):
        if training:
            return inputs * self.var
        else:
            return inputs

    def get_config(self):
        return {'a': self.var.numpy()}

    # there is actually no need to define 'from_config' here, since returning cls(**config) is the default behavior

    @classmethod
    def from_config(cls, config):
        return cls(**config)


layer = CustomLayers(5)
layer.var.assign(2)

serialized_layer = keras.layers.serialize(layer)
new_layer = keras.layers.deserialize(
    serialized_layer, custom_objects={'CustomLayer': CustomLayers}
)

# Registering the custom obj
# keras keeps a note of which class generated the config.
# above tf.keras.layers.serialize
# {'class_name': 'CustomLayer', 'config': {'a': 2}}


# custom layer and function example
class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.untis),
            initializer='random_normal',
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer='random_normal', trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({'units': self.units})
        return config

def custom_activation(x):
    return tf.nn.tanh(x) ** 2

# Make a model with the CustomLayer and custom_activation
inputs = keras.Input((32,))
x = CustomLayer(32)(inputs)
outputs = keras.layers.Activiation(custom_activation)(x)
model = keras.Model(inputs, outputs)

# Retrive the config
config = model.get_config()

# At loading time, register the custom objects with a 'custom_object_scope':
custom_objects = {'CustomLayer': CustomLayer, 'custom_activation': custom_activation}
with keras.utils.custom_objects_scope(custom_objects):
    new_model = keras.Model.from_config(config)

# also possible ot do in-memory cloning: tf.keras.models.clone_model()
# this is equivalent to getting he config then recreating the model from its config
# (does not preserve compilation or weights)

#Transfering weights from one layer to another in memory

def create_layer():
    layer = keras.layers.Dense(64, activation='relu', name='dense_2')
    layer.build((None, 784))
    return layer

layer_1 = create_layer()
layer_2 = create_layer()

# copy weights from layer1 to layer 2
layer_2.set_weights(layer_1)


# Transfering weights from one model to another with compatible architecture

# Creat a simple functional model
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
functional_model = keras.Model(imputs=inputs, outputs=outputs, name='3_layer_mlp')

# Define a subclassed model with the same architecture
class SubclassedModel(keras.Model):
    def __init__(self, output_dim, name=None):
        super(SubclassedModel, self).__init__(name=name)
        self.output_dim = output_dim
        self.dense_1 = keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = keras.layers.Dense(64, activation='relu', name='dense_2')
        self.dense_3 = keras.layers.Dense(10, name='predictions')(x)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

    def get_config(self):
        return {'output_dim': self.output_dim, 'name': self.name}


subclassed_model = SubclassedModel(10)
# Call the subclassed model once to create the weights.
subclassed_model(tf.ones((1, 784)))


# copy weights from functional_model to subclassed_model
subclassed_model.set_weights(functional_model.get_weights())

assert len(functional_model.weights) == len(subclassed_model.weights)
for a, b in zip(functional_model.weights, subclassed_model.weights):
    np.testing.assert_allclose(a.numpy, b.numpy())


# the case of stateless layers
# stateless layers do not change the order or number of weights, ergo models can have compatible architectures
# even if there are extra/missing stateless layers

inputs = keras.Input(shape=(784,), name='digits')
x = keras.kayers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.kayers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
functional_model=keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

inputs = keras.Input(shape=(784,), name="digits")
x = keras.kayers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.kayers.Dense(64, activation='relu', name='dense_2')(x)

# Add a dropout layer, which does not contain any weights
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
functional_model_with_dropout = keras.Model(
    inputs=inputs, outputs=outputs, name='3_layer_mlp'
)

functional_model_with_dropout.set_weights(functional_model.get_weights())

## TF checkpoint format
# Runnable example

sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name='digits'),
        keras.layers.Dense(64, activation='relu', name='dense_1'),
        keras.layers.Dense(64, activation='relu', name='dense_2'),
        keras.layer.Dense(10, name='predictions')
    ]
)

sequential_model.save_weights('ckpt')
load_status = sequential_model.load_weights('ckpt')

# 'assert_consumed' can be used as validation that all variable values have been
# restored from the checkpoint. See 'tf.train.Checkpoint.restore' for other
# methods in the status object

load_status.assert_consumed()

class CustomLayer(keras.layers.Layer):
    def __init__(self, a):
        self.var = tf.Variable(a, name='var_a')

    layer = CustomLayer(5)
    layer_ckpt = tf.train.Checkpoint(layer=layer).save('custom_layer')

    ckpt_reader = tf.train.load_checkpoint(layer_ckpt)
    ckpt_reader.get_variable_to_dtype_map()


## Transfer learning example, if two models have same architechure they can share checkpoints

inputs = keras.Input(shape=(784,), name='digits')
x = keras.kayers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.kayers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
functional_model=keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

# Extract portion of functional model defined in setup section
# The following lines produe a new model that excludes the final output
# layer of the functional model

pretrained = keras.Model(
    functional_model.inputs, functional_model.layers[-1].input, name='pretrined_model'
)

# Randomly assign 'trained' weights
for w in pretrained.weights:
    w.assign(tf.random.normal(w.shape))
pretrained.save_weights('pretrianed_ckpt')
pretrained.summary()


## Assume this is a separate programe where only 'pretrained_ckpt' exists.
# create a new functional model with a different output dimension

inputs = keras.Input(shape=(784,), name='digits')
x = keras.kayers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.kayers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
functional_model=keras.Model(inputs=inputs, outputs=outputs, name='new_model')

# Load weights from pretraind_ckpt intp model
model.load_weights('pretrainded_ckpt')

# check that all of the pretrained weights have been loaded
for a, b in zip('pretrained_ckpt'):
    np.testing.assert_allclose(a.numpy(), b.numpy())

print('\n', '-' * 50)
model.summary()

# Example2 Sequential model
# recreate the pretraind model, and load the saved weights
inputs = keras.Input(shape=(784,), name='digits')
x = keras.kayers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.kayers.Dense(64, activation='relu', name='dense_2')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
pretrained_model=keras.Model(inputs=inputs, outputs=outputs, name='pretrained')

# Sequential
model = keras.Sequential([pretrained_model, keras.layers.Dense(5, name='predictions')])
print(model.summary())

pretrained_model.load_weights('pretrained_ckpt')

# Warning! Calling `model.load_weights('pretrained_ckpt')` won't throw an error,
# but will *not* work as expected. If you inspect the weights, you'll see that
# none of the weights will have loaded. `pretrained_model.load_weights()` is the
# correct method to call.


# The next question is, how can weights be saved and loaded to different models if the model architectures are quite different?
# The solution is to use tf.train.Checkpoint to save and restore the exact layers/variables.
first_dense = functional_model.layers[1]
last_dense = functional_model.layers[-1]
ckpt_path = tf.train.Checkpoint(
    dense=first_dense, kernel=last_dense, bias=last_dense.bias
).save('ckpt')

# define the subclassed model
class ContrivedModel(keras.Model):
    def __init__(self):
        super(ContrivedModel, self).__init__()
        self.first_dense = keras.layers.Dense(64)
        self.kernel = self.add_variable('kernal', shape=(64, 10))
        self.bias = self.add_variable('bias', shape=(10,))

    def call(self, inputs):
        x = self.first_dense(inputs)
        return tf.matmul(x, self.kernel) + self.bias


model = ContrivedModel()
# Call model on inputs to create the variables of the dense layer
_ = model(tf.ones((1, 784)))

# Create a checkpoint with the same structure as before, and load the weights
tf.train.Checkpoint(
    dense=model.first_dense, kernel=model.kernal, bias=model.bias
).restore(ckpt_path).assert_consumed()

# model can use a hdf5 checkpoint if it has the same layers and trainable statuses as saved in the checkpoint.
# runnable example

sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions")
    ]
)
sequential_model.save_weights('weights.h5')
sequential_model.load_weights('weights.h5')


# changing layer.trainable may result in a differnt layer weights ording when the model contains nested layers

class NestedDenseLayer(keras.layers.Layer):
    def __init__(self, units, name=None):
        super(NestedDenseLayer, self).__init__()
        self.dense_1 = keras.layers.Dense(units, name='dense_1')
        self.dense_2 = keras.layers.Dense(units, name='dense_2')

    def call(self, inputs):
        return self.dense_2(self.dense_1(inputs))

nested_model = keras.Sequential([keras.Input((784,)), NestedDenseLayer(10, 'nested')])
variable_names = [v.name for v in nested_model.weights]
print('variables: {}'.format(variable_names))

print('"\nChanging trainable status of one of the nested layers..."')
nested_model.get_layers('nested').dense_1.trainable = False

variable_names_2 = [v.name for v in nested_model.weights]
print('\nvariables: {}'.format(variable_names_2))
print('variable ording changed:', variable_names != variable_names_2)



# Transfer learning with HDF5
# it is recommended to load the weights into the original checkpointed
# model, and then extract the desired weights/layers into a new model.


def create_functional_model():
    inputs = keras.Input(shape=(784,), name='digits')
    x = keras.kayers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = keras.kayers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = keras.layers.Dense(10, name='predictions')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')


    functional_model = create_functional_model()
    functional_model.save_weights('pretrained_weights.h5')

    # in seperate program hypothetically
    pretrained_model = create_functional_model()
    pretrianed_model.load_weights('pretrained_weights.h5')

    # create a new model by extracting layers from the origonal model:
    extracted_layers = pretrained_model.layers[:-1]
    extracted_layers.append(keras.layers.Dense(5, name='dense_3'))
    model = keras.Sequential(extracted_layers)
    print(model.summary())

