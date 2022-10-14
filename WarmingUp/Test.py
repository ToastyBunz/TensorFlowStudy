import json
import tensorflow as tf
import numpy as np
import urllib
# from tf.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


def solution():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasim.json')

    # Do not change this code
    vocab_size = 1000
    embedding_dim = 16
    max_len = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    training_size = 20000

    sentances = []
    labels = []

    # Time for Nathan to show his fire
    with open("S:\Python\Code\TensorReview\TF_Questions\sarcasim.json", "r") as f:
        datastore = json.load(f)


    headlines = []
    labels = []

    # print(datastore)
    for item in datastore:
        headlines.append(item['headline'])
        labels.append(item['is_sarcastic'])

    # print(labels)
    trianing_headlines = headlines[0:training_size]
    testing_headlines = headlines[training_size:]
    trianing_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # initialize tokenizer, set number of vocab size, set oov token
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(trianing_headlines)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(trianing_headlines)
    training_padded = tf.keras.utils.pad_sequences(training_sequences,
                                                   maxlen=max_len,
                                                   padding=padding_type,
                                                   truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_headlines)
    testing_padded = tf.keras.utils.pad_sequences(testing_sequences,
                                                  maxlen=max_len,
                                                  padding=padding_type,
                                                  truncating=trunc_type)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-4,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='mymodel.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    training_padded = np.array(training_padded)
    training_labels = np.array(trianing_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    # Model Time
    model = tf.keras.Sequential([
        # tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        # tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(30, activation='relu'),
        # tf.keras.layers.Dense(24, activation='relu'),
        # tf.keras.layers.Dense(1, activation='sigmoid')

        # Stolen but cool code
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')


    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(training_padded,
              training_labels,
              epochs=10,
              validation_data=(testing_padded, testing_labels),
              verbose=2)

    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        tf.keras.layers.Dense(1, activation='sigmoid')

    ])

    export_model.compile(
        loss= tf.losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']
    )

    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)


    examples = [
      "The movie was great!",
      "The movie was okay.",
      "The movie was terrible..."
    ]

    export_model.predict(examples)

    return model
solution()