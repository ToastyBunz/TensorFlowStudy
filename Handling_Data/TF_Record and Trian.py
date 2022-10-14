import tensorflow as tf
import numpy as np
import IPython.display as display

# the following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a strin / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# examples of how these functions work. Note the varying input types and standardized output type.
# if input type for a function does not match one of the coercible types stated above the function will raise an
# exception

print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'tes_bytes'.encode('utf=8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))

feature = _float_feature(np.exp(1))

feature.SerializeToString()

# number of observations in the dataset
n_observations = int(1e4)
# Boolean feature, encoded as False or True
feature0 = np.random.choice([False, True], n_observations)
# Integer feature, random from 0 to 4
feature1 = np.random.randint(0, 5, n_observations)

# string feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)

# each of these features cna be coerced into a tf.train.Example

def serialize_example(feature0, feature1, feature2, feature3):
    """ Creates a tf.train.Example message ready to be a written to a file"""

    # Create a dictionary mapping the feature name to the tf.train.Example-compatiable
    # data type
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3)
    }

    # Create a Features message using tf.train.Example

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# this is an example observation from the dataset
example_observation = []
serialize_example = serialize_example(False, 4, b'goat', 0.9876)
print(serialize_example)

example_proto = tf.train.Example.FromString(serialize_example)
print(example_proto)

# records are concatedated together into a CRC (Cyclic redundancy check)
# masked_crc = ((crc >> 15 ) | (crc << 17)) + oxa282ead8ul

#  TF recordid files using tf.data
# writing tfrecord file easiest way to get data into a dataset is to use from_tensor_slices method

tf.data.Dataset.from_tensor_slices(feature1)
# applied to a tuple of arrays it returnes a dataset of tuples:
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
features_dataset

for f0, f1, f2, f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)

# tf.data.Dataset.map method to apply a function to each element of a Dataset
# but the mapped fucntion must operate in the TensorFlow graph mode
# it must operate on and return tf.tensors. BUT A non-tensor functino like serialize_example can be wrapped by
# tf.py_function and requires to specify the shape and type of information that is otherwise unavailable

def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3), # pass these args to the above function (serialize_example)
        tf.string) # the return type is 'tf.string
    return tf.reshape(tf_string, ()) # the result is a scalar

tf_serialize_example(f0, f1, f2, f3)

# apply this function to every element in the dataset
serialized_features_dataset = features_dataset.map(tf_serialize_example)
print(serialized_features_dataset)

def generator():
    for features in features_dataset:
        yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=()
)

serialized_features_dataset
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

# can also read teh TFRecord file using tf.data.TfRecordDataset
# using TFRecordDatase can be useful for sandardizing input data and optimizing performance

filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset

for raw_record in raw_dataset.take(10):
    print(repr(raw_record))

# the tensors just printed can be parsed (shown below). NOTE feature description is necessary here because
# tf.data.Dataset uses graph execution and needs the description to build their shape and type signature

# Create a description of the features.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.int32, default_value=0.0)
}

def _parse_function(example_photo):
    # Parse the input 'tf.train.Example' proto using the dict above
    return tf.io.parse_single_example(example_proto, feature_description)

# Alternatively use tf.parse_example to parse the whole batch at once. Apply this funciton to each item in the dataset
# using the tf.data.Dataset.map

parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)

# eager_execution will display the observations in the dataset. These are 10,000 observations inthe dataset but
# you will only display first 10. each item is a tf.Tensor and numpy element displays the value of the feature:

for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))

# write the 10,000 observations to the file test.tfrecord. Observation > tf.train.Example message > written to file

# write the 'tf.train.Example 'observations' to the file.
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

# Reading a TFRecord file. these serialized tensors can be sasily parsed using tf.train.Example.ParseFromString
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset


for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
# this will return a tf.train.Example proto with is difficult to use as is but is fundamentally like:

# Dict[str, Union[List[float], List[int], List[str]]]

# next block manually converts the Example to a dict of numpy arrays w/o using TF Ops. More info:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/feature.proto

result = {}
# example.features.features is the dictionary
for key, feature in example.features.features.items():
    # the values are teh Feature objects which contain a 'kind' which contains:
    # one of three fields: bytes_list, float_list, int64_list
    kind = feature.WhichOneof('kind')
    result[key] = np.array(getattr(feature, kind).value)

print(result)

