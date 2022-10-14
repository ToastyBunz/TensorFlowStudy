import collections
import pathlib

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import losses
from keras import utils
from keras.layers import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

# three different english translations of the same work, Illiad. Identify translator

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = pathlib.Path(text_dir).parent
list(parent_dir.iterdir())

# in this example create a dataset form txt file where each example is a line from the text
# text line dataset is usefule

# iterate through tese files, loading each one into its own dataset. each one needs a label. use dataset.map
# will iterate every exaple returning (example, label) pars

def labeler(example, index):
    return example, tf.cast(index, tf.int64)

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(str(parent_dir/file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False
)

for text, label in all_labeled_data.take(10):
    print('Sentences: ', text.numpy())
    print('Label: ', label.numpy())

# instead of TextVectorization here use Text apis StaticBocabularyTable to map tokens to integers
# convert the text to lower case and tokenize

tokenizer = tf_text.UnicodeScriptTokenizer()

def tokenize(text, unused_label):
    lower_case = tf_text.case_fold_utf8(text)
    return tokenizer.tokenize(lower_case)

tokenized_ds = all_labeled_data.map(tokenize)
for text_batch in tokenized_ds.take(5):
    print('Tokens: ', text_batch.numpy())

AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

tokenized_ds = configure_dataset(tokenized_ds)

vocab_dict = collections.defaultdict(lambda : 0)
for toks in tokenized_ds.as_numpy_iterator():
    for tok in toks:
        vocab_dict[tok] += 1

VOCAB_SIZE = 10000

vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = [token for token, count in vocab]
vocab = vocab[:VOCAB_SIZE]
vocab_size = len(vocab)
print('vocab size: ', vocab_size)
print('First five vocab entries: ', vocab[:5])


keys = vocab
values = range(2, len(vocab) + 2) # reserve 0 for padding and 1 for oov tokens
init = tf.lookup.KeyValueTensorInitializer(
     keys, values, key_dtype=tf.string, value_dtype=tf.int64
 )

num_oov_buckets = 1
vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)

#finally standardize token and vect
def preprocess_text(text, label):
    standardized = tf_text.case_fold_utf8(text)
    tokenized = tokenizer.tokenize(standardized)
    vectorized = vocab_table.lookup(tokenized)
    return vectorized, label

# try on single example and print output
example_text, example_label = next(iter(all_labeled_data))
print('Sentence: ', example_text.numpy())
vectorized_text, example_label = preprocess_text(example_text, example_label)
print('Vectorized sentences: ', vectorized_text.numpy())

# now run the preprocess function on the dataset
all_encoded_data = all_labeled_data.map(preprocess_text)

train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
validation_data = all_encoded_data.take(VALIDATION_SIZE)

train_data = train_data.padded_batch(BATCH_SIZE)
validation_data = validation_data.padded_batch(BATCH_SIZE)

sample_text, sample_labels = next(iter(validation_data))
print('Text batch shape: ', sample_text.shape)
print('Label batch shape: ', sample_labels.shape)
print('First text example: ', sample_text[0])
print('first label examle: ', sample_labels[0])

# 0 = padding, 1 = oov
vocab_size += 2
train_data = configure_dataset(train_data)
validation_data = configure_dataset(validation_data)

def create_model(vocab_size, num_labels):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(num_labels)
    ])
    return model


# Training
model = create_model(vocab_size=vocab_size, num_labels=3)

model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

loss, accuracy = model.evaluate(validation_data)
print('Loss: ', loss)
print('Accuracy: {:2.2}'.format(accuracy))

# Export model (allow raw string to be tested)
MAX_SEQUENCE_LENGTH = 250

preprocess_layer = TextVectorization(
    max_tokens=vocab_size,
    standardize=tf_text.case_fold_utf8,
    split=tokenizer.tokenize,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

preprocess_layer.set_vocabulary(vocab)

export_model = tf.keras.Sequential([
    preprocess_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)

# create a test dataset for raw strings
test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
test_ds = configure_dataset(test_ds)

loss, accuracy = export_model.evaluate(test_ds)
print('loss: ', loss)
print('accuracy: {:2.2%}'.format(accuracy))

inputs = [
    "Join'd to th' Ionians with their flowing robes,", # label 1
    "the allies, and his armor flashed about his so that he seemed to all", # Label 2
    "And with loud clangor of his arms he feel" # label 0
]

predicted_scores = export_model.predict(inputs)
predicted_labels = tf.math.argmax(predicted_scores, axis=1)

for input, label in zip(inputs, predicted_labels):
    print("Question: ", input)
    print("Predicted label: ", label.numpy())


# Download mre datasets

train_ds = tfds.load(
    'imdb_reviews',
    split='train[:80%]',
    batch_size=BATCH_SIZE,
    shuffle_files=True,
    as_supervised=True
)

val_ds = tfds.load(
    'imdb_reviews',
    split='train[80%:]',
    batch_size=BATCH_SIZE,
    shuffle_files=True,
    as_supervised=True
)

for review_batch, label_batch in val_ds.take(1):
    for i in range(5):
        print("review: ", review_batch[i].numpy())
        print("label: ", label_batch[i].numpy())

vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

# txt only dataset (no lables)
train_text = train_ds.map(lambda text, labels: text)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)

# config dataset for preformance
train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)

model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=1)
model.summary()

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(train_ds, validation_data=val_ds, epochs=3)

loss, accuracy = model.evaluate(val_ds)
print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy']
)

# 0 negative review
# 1 positive review

inputs = [
    "This is a fantastic movie.",
    "This is a bad movie.",
    "This movie was so bad that it was good.",
    "I will never say yes to watching this movie.",
]

predicted_scores = export_model.predict(inputs)
predicted_labels = [int(round(x[0])) for x in predicted_scores]

for input, label in zip(inputs, predicted_labels):
    print("Question: ", inputs)
    print("Predicted label: ", label)

