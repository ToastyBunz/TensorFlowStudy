import tensorflow as tf
import IPython.display as display


# end to end image data (tf Records)
# able to preprocess data then run in multiple models. raw data doesn't need to be reloaded everytime

cat_in_snow  = tf.keras.utils.get_file(
    '320px-Felis_catus-cat_on_snow.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')

williamsburg_bridge = tf.keras.utils.get_file(
    '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))

display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))


# this doesnt really display the images I will need to find another way to visually inspect them


# as before encode the features with tf.train.Example, this stores raw string feature along with H W D and label
image_labels = {
    cat_in_snow: 0,
    williamsburg_bridge: 1
}

# this example just using cat
image_string = open(cat_in_snow, 'rb').read()
label = image_labels[cat_in_snow]

# Create a dict with features that may be relevant
def image_example(image_string, label):
    image_shape = tf.io.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_features(image_string)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string,label)).split('\n')[:15]:
    print(line)

print('...')

# Write the raw image files to 'images.tfrecords'
# process the two images into 'tf.train.Example' messages
# then write to a '.tfrecords' file
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())

# I now have a images.tfrecords I can iterate of the records to read back the images.
# extract the image

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dict describing the features
image_features_description = {
    'height': tf.io.FixedLenFeature([], tf.init64),
    'width': tf.io.FixedLenFeature([], tf.init64),
    'depth': tf.io.FixedLenFeature([], tf.init64),
    'label': tf.io.FixedLenFeature([], tf.init64),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}

def _parse_image_function(example_proto):
    # parse input using dict above
    return tf.io.parse_single_example(example_proto, image_features_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset)

for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    display.display(display.Image(data=image_raw))