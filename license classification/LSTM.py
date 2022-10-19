import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

license_dataset = pd.read_csv('dataset/license_sample.csv')

vocab_size = 5000 # make the top list of words (common words)
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # OOV = Out of Vocabulary
training_portion = .8

license_text = []
license_name = []

with open("dataset/license_sample.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        license_name.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        license_text.append(article)

sample_license_view = license_text[4]

print(len(license_name))
print(len(license_text))

# create
train_size = int(len(license_text) * training_portion)

train_articles = license_text[0: train_size]
train_labels = license_name[0: train_size]

validation_articles = license_text[train_size:]
validation_labels = license_name[train_size:]

print("train_size :", train_size)
print(f"train_articles : {len(train_articles)}")
print("train_labels :", len(train_labels))
print("validation_articles: ", len(validation_articles))
print("validation_labels: ", len(validation_labels))

# tokenizer and convert to sequence example:
# tokenizer - assigns each word, a number in incremental fashion
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(['Hi my name is akshay'])
tokenizer.word_index

text_sequence = tokenizer.texts_to_sequences(['Hi my name is aki'])

# example ended

# apply the same below

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
tokenizer_indices = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)

train_sequences[7]

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train_padded[7]

# follow same for validation test
validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# look at labels
print(set(license_name))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(license_name)
label_tokenizer.word_index

training_label_seq = label_tokenizer.texts_to_sequences(train_labels)
validation_label_seq = label_tokenizer.texts_to_sequences(validation_labels)



# create model

model = Sequential()
model.add(Embedding(vocab_size,embedding_dim))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(6,activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

tf.convert_to_tensor(training_label_seq)
arr = np.array(training_label_seq)

training_label_seq = np.array(training_label_seq, dtype = None, copy = True, order = None, subok = False, ndmin = 0)

history = model.fit(train_padded, training_label_seq, epochs=20, validation_data=(validation_padded, validation_label_seq), verbose=2)

type(validation_label_seq)
