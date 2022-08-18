import json
import random
import re
import string
import time
import os
import numpy as np
# Preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
# For Vectorization
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from sklearn.utils import compute_class_weight
# building model
import tensorflow as tf
import tensorflow.python.keras.optimizer_v1
from tensorflow import keras
from keras.layers import Dense, Activation, Embedding, GlobalMaxPool1D, Dropout, Conv1D, Conv2D, LSTM, Flatten, BatchNormalization
from keras.models import Sequential
from keras import Input
from tensorflow.python.keras.optimizer_v1 import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold, StratifiedKFold
# metrics
from keras.metrics import Precision, Recall
from keras import backend as K
from sklearn.metrics import average_precision_score
# plot
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
# optuna
import optuna
from optuna import trial
# keras tuner
import keras_tuner
from keras import layers
from keras_tuner import RandomSearch


dataset_file_path = 'top_40_labels_dataset.json'
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


# plot word distribution of policy_texts
def plot_word_distri(sentence_list):
    word_length = []
    for sentence in sentence_list:
        words = sentence.split()
        senten_len = len(words)
        word_length.append(senten_len)

    plt.figure(figsize=(12, 6))
    values = word_length
    plt.title('Word distribution of the policy text')
    plt.grid()
    plt.bar(range(len(sentence_list)), values)
    plt.xlabel(f'Number of policy text: {len(sentence_list)}')
    plt.ylabel('No. of words in a policy text')
    plt.show()

# ----- Preprocessing -----
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


# map the function to the dataset to preprocess the text
print('--preprocessing data--')
preprocessed_text = list(map(lambda text: preprocess_text(text), X))
print('--preprocessing done--')

# plot word distribution of preprocessed policy texts
def plot_prepro_word_distri(pre_text):
    word_length = []
    for senten in pre_text:
        words = senten.split()
        senten_len = len(words)
        word_length.append(senten_len)

    plt.figure(figsize=(12, 6))
    values = word_length
    plt.title('Word distribution of the policy text')
    plt.grid()
    plt.bar(range(len(pre_text)), values)
    plt.xlabel(f'Number of policy text: {len(pre_text)}')
    plt.ylabel('No. of words in a preprocessed policy text')
    plt.show()

# --------- build the word2vec model ---------
# tokenize the the policy texts
tokenized_data = []
for sentence in preprocessed_text:
    sentc_tokens = word_tokenize(sentence)
    tokenized_data.append(sentc_tokens)

# title = re.sub('\D', '', dataset_file_path)
# word2vec_model_file = 'word2vec_top'+ f'{title}' +'.model'
start = time.time()
w2v_model = Word2Vec(sentences=tokenized_data, size=300, window=10, min_count=1)
print("Time taken to train word2vec model: ", round(time.time()-start, 0), 'seconds')
# w2v_model.save(word2vec_model_file)

w2v_model.train(tokenized_data, epochs=10, total_examples=len(tokenized_data))

vocab = w2v_model.wv.vocab
print("The total words : ", len(vocab))

# create a dictionary with word and key as the embedding (used for creating embed matrix)
vocab = list(vocab.keys())
word_vect_dict = {}

for word in vocab:
    word_vect_dict[word] = w2v_model.wv.get_vector(word)
# print(f'key-value pair entries: {len(word_vect_dict)}')

# encode the text and define parameters
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_text)

# find the maximum length = 214
def get_max_len(encoded_text):
    len_all_ptext = []
    for encode in encoded_text:
        len_all_ptext.append(len(encode))
    max_len = max(len_all_ptext)
    return max_len


vocab_size = len(tokenizer.word_index) + 1 # total vocabulary size
encoded_ptext = tokenizer.texts_to_sequences(preprocessed_text) # encoded policy texts
maxlen = 215 # maximum length of each policy text
embed_dim = 300

# pad the sequence of policy text
ptext_pad = pad_sequences(encoded_ptext, maxlen=maxlen, padding='post')

# creating the embedding matrix
embedding_matrix = np.zeros(shape=(vocab_size, embed_dim))
hits = 0
misses = 0

for word, i in tokenizer.word_index.items():
    embed_vector = word_vect_dict.get(word)
    try:
        if embed_vector is not None:
            embedding_matrix[i] = embed_vector
            hits += 1
        else:
            misses += 1
    except:
        pass

print(f'converted words {hits} ({misses} missed words)')

# split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(ptext_pad, y, test_size=0.2, random_state=42)

# ------------ binarize the multiple labels ------------
mlb = MultiLabelBinarizer()
y_train_mlb = mlb.fit_transform(y_train)
y_test_mlb = mlb.transform(y_test)
label_classes = mlb.classes_

# create dictionary counters with the key as the label name and the value as the total number of labels
counters = {}
for labels in y:
    for label in labels:
        if counters.get(label) is not None:
            counters[label] += 1
        else:
            counters[label] = 1


# plot bar chart to visualize the distribution of labels (check for imbalance)
def plot_class_distribution2(count):
    plt.figure(figsize=(12, 6))
    values = list(count.values())
    name = list(count.keys())
    plt.title('All labels: Distribution of the no. of times a label appeared in a policy text')
    plt.grid()
    plt.bar(range(len(count)), values)
    plt.xlabel('Label numbers. (All 36 labels)')
    plt.ylabel('No. of times each label appeared in the whole dataset')
    plt.show()


# calculates class weights for label due to label imbalance
class_weights = {}
for index, label in enumerate(label_classes):
    class_weights[index] = len(y) / counters.get(label)


# --- custom metrics ---
def c_recall(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def c_precision(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def c_f1(y_test, y_pred):
    precision = c_precision(y_test, y_pred)
    recall = c_recall(y_test, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# keras-tuner for optimization
precision = Precision()
recall = Recall()

def tuner_model(hp):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=maxlen, weights=[embedding_matrix]))
    model.add(Dropout(0.6))
    model.add(Conv1D(filters=hp.Int(f'filters', min_value=32, max_value=512, step=16),
                     kernel_size=hp.Int(f'kernel_size', min_value=3, max_value=11, step=2),
                     padding='same',
                     activation='relu'))

    model.add(Dropout(0.6))
    model.add(GlobalMaxPool1D())
    model.add(Flatten())
    model.add(Dense(36))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=[precision, recall, c_f1])

    return model


tuner = RandomSearch(
    tuner_model,
    objective=keras_tuner.Objective('val_c_f1', direction='max'),
    max_trials=3,
    executions_per_trial=3,
    overwrite=True)

tuner.search_space_summary()

tuner.search(X_train, y_train_mlb,
             epochs=5,
             validation_data=(X_test, y_test_mlb))

best_model = tuner.get_best_models()

best_models = tuner.get_best_models(num_models=2)

# -----------------------------------------------------------------------------
# get optimal hp
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# train the model
model = tuner.hypermodel.build(best_hps)

history = model.fit(X_train, y_train_mlb, epochs=20, validation_split=0.2)

val_f1_per_epoch = history.history['val_c_f1']
best_epoch = val_f1_per_epoch.index(max(val_f1_per_epoch)) + 1
print('Best-f1 epoch: %d' % (best_epoch,))

# evaluate the result








