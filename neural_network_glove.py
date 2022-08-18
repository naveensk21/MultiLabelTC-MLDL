import json
import random
import re
import string
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import time
import tqdm

# Preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# For Vectorizing
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from ast import literal_eval

# building model
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from tensorflow.python.keras.optimizers import adam_v2
from keras.metrics import Precision

from tensorflow.python.keras import layers
from tensorflow import keras
import tensorflow as tf

# ------------------------
from numpy import array
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
# --------------------------


from sklearn.model_selection import train_test_split
from ast import literal_eval


dataset_file_path = 'top_20_labels_dataset.json'
with open(dataset_file_path) as fp:
    dataset_json = json.load(fp)

def create_x_y():

    with open(dataset_file_path) as fp:
        dataset_json = json.load(fp)

    x = [] # policy texts
    y = [] # labels

    random.shuffle(dataset_json)
    for datapoint in dataset_json:
        x.append(datapoint['policy_text'])
        y.append(datapoint['labels'])

    # print(f"Loaded {len(x)} policies with {len(y)} corresponding sets of policy practices")
    return x, y


# create x and y
X, y = create_x_y()

# with open('top_40_labels.json') as fp:
#     top_40_labels = json.load(fp)
# print(len(top_40_labels))

labels = 36

# ----- Preprocessing -----
# function to strip the html tags
def preprocess_text(text):
    text = text.lower()
    text = re.sub(re.compile('<.*?>'), '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    word_tokens = text.split()
    le=WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    word_tokens = [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    cleaned_text = " ".join(word_tokens)

    return cleaned_text


# map the function to the dataset to strip html tags
clean_data = list(map(lambda text: preprocess_text(text), X))
# print('Before Cleaning: \n', X[1])
# print('After Cleaning: \n', clean_data[1])


# split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, test_size=0.2, random_state=42)


# binarize the labels
mlb = MultiLabelBinarizer()
y_train_mlb = mlb.fit_transform(y_train)
y_test_mlb = mlb.transform(y_test)
label_classes = mlb.classes_

# word embedding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
max_len = 215

X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

# glove filepath (use glove embedding to convert the text into numerical representation)
# glove = open('dataset/glove.6B.100d.txt', encoding='utf8')


# def build_embedding_matrix(glove, word_index, embedding_dimension):
#     embedding_matrix_glove = np.zeros((vocab_size, embedding_dimension))
#
#     for line in glove:
#         word, *vector = line.split()
#         if word in word_index:
#             indx = word_index[word]
#             embedding_matrix_glove[indx] = np.array(vector, dtype=np.float32)[:embedding_dimension]
#     return embedding_matrix_glove
#
#
embed_dim = 100
# embedding_matrix = build_embedding_matrix(glove, tokenizer.word_index, embed_dim)
embed_vector_len = 100


def construct_embed_matrix(word_index):
    embed_dict = dict()
    with open('dataset/glove/glove.6B.100d.txt', 'r', errors='ignore', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = ''.join(values[:-300])
            if word in word_index.keys():
                # get vector
                vector = np.asarray(values[-300:], 'float32')
                embed_dict[word] = vector

    num_words = len(word_index) + 1

    embedding_matrix = np.zeros((num_words, embed_vector_len))

    for word, i in tqdm.tqdm(word_index.items()):
        if i < num_words:
            vect = embed_dict.get(word, [])
            if len(vect)> 0:
                embedding_matrix[i] = vect[:embed_vector_len]
    return embedding_matrix

embedding_matrix = construct_embed_matrix(tokenizer.word_index)


def build_cnn_model():
    n_classes = len(mlb.classes_)

    model = Sequential()

    model.add(Embedding(vocab_size, embed_dim, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(Dropout(0.6))
    model.add(Conv1D(350, 4, padding='valid', activation='relu', strides=1))
    model.add(Dropout(0.6))
    model.add(Conv1D(370, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Flatten())
    model.add(Dense(n_classes))
    model.add(Activation('sigmoid'))

    return model


# params
start = time.time()
epoch = 20
# 19-0.01, 17-0.11(high recall),
batch_size = 16
lr = 0.001
opt = keras.optimizers.Adam(learning_rate=lr)

model = build_cnn_model()

model.summary()

# compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

# fit the model
history = model.fit(X_train, y_train_mlb, epochs=epoch, batch_size=batch_size, validation_split=0.1, shuffle=True)

# display scores
print("Time taken to fit the model: ", round(time.time()-start, 0), 'seconds')
score = model.evaluate(X_test, y_test_mlb)
print(f'{model.metrics_names[0]}: {score[0]}')
print(f'{model.metrics_names[1]}: {score[1]}')
print(f'{model.metrics_names[2]}: {score[2]}')





