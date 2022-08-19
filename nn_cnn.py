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

############### Build model ################
def build_cnn_model():
    n_classes = len(mlb.classes_)

    model = Sequential()

    model.add(Embedding(vocab_size, embed_dim, input_length=maxlen, weights=[embedding_matrix]))
    model.add(Dropout(0.6))
    model.add(Conv1D(350, 4, padding='valid', activation='relu', strides=1))
    model.add(Dropout(0.6))
    model.add(Conv1D(370, 3, padding='valid', activation='relu', strides=1))
    # model.add(Dropout(0.6))
    # model.add(Conv1D(165, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Flatten())
    model.add(Dense(n_classes))
    model.add(Activation('sigmoid'))

    return model

"""
# # bs-16, lr-0.001, dp-0.6-0.6, conv-160,4
loss: 0.15778566896915436
precision: 0.5239361524581909
recall: 0.5596590638160706
c_f1: 0.5378256440162659
-------------------------
loss: 0.14787784218788147
precision: 0.5360000133514404
recall: 0.5775862336158752
c_f1: 0.5729897022247314
--------------------------

#bs-16, lr-0.01, dp-0.5-0.5, conv1-250,3, conv1-270,3
loss: 0.10036792606115341
precision: 0.6052631735801697
recall: 0.5154061913490295
c_f1: 0.5591225624084473

#bs-16, lr-0.01, dp-0.5-0.5, conv1-220,3, conv1-220,4
loss: 0.0927027016878128
precision: 0.5697329640388489
recall: 0.5714285969734192
c_f1: 0.5692447423934937

#bs-16, lr-0.01, dp-0.5-0.5, conv1-220,3, conv1-220,3
loss: 0.10056645423173904
precision: 0.5894736647605896
recall: 0.6363636255264282
c_f1: 0.6027687191963196

# bs-16, lr-0.001, dp-0.6-0.6, conv-350,3, conv-370,3
loss: 0.10055424273014069
precision: 0.53125
recall: 0.571865439414978
c_f1: 0.5578574538230896


#bs-16, lr-0.001, dp-0.6-0.6, conv-183,3
loss: 0.29696816205978394
precision: 0.3586800694465637
recall: 0.6983240246772766
c_f1: 0.47736337780952454


"""
# params
start = time.time()
epoch = 30
# 19-0.01, 17-0.11(high recall),
batch_size = 16
lr = 0.001
opt = keras.optimizers.Adam(learning_rate=lr)

model = build_cnn_model()

model.summary()

# compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.Precision(), keras.metrics.Recall(), c_f1])

# fit the model
history = model.fit(X_train, y_train_mlb, epochs=epoch, batch_size=batch_size, validation_split=0.1, shuffle=True)

# display scores
print("Time taken to fit the model: ", round(time.time()-start, 0), 'seconds')
score = model.evaluate(X_test, y_test_mlb)
print(f'{model.metrics_names[0]}: {score[0]}')
print(f'{model.metrics_names[1]}: {score[1]}')
print(f'{model.metrics_names[2]}: {score[2]}')
print(f'{model.metrics_names[3]}: {score[3]}')

exit()

################## K-fold ####################
def kfold_val():
    cross_val = KFold(n_splits=5, shuffle=True, random_state=42)
    # list of val scores
    val_loss = []
    val_precision = []
    val_recall = []
    fld_score = []
    # params
    lr = 0.01
    opt = keras.optimizers.Adam(learning_rate=lr)
    fold_num = 1

    for train_i, test_i in cross_val.split(X_train, y_train_mlb):
        keras.backend.clear_session()

        # generate print
        print('------------------------------------------')
        print(f'Training for fold {fold_num}....')
        print('------------------------------------------')

        x_train_val, x_test_val = X_train[train_i], X_train[test_i]
        y_train_val, y_test_val = y_train_mlb[train_i], y_train_mlb[test_i]

        model = build_cnn_model()

        rdc_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-06, verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto",
                                   baseline=None,
                                   restore_best_weights=True)

        model.compile(loss='binary_crossentropy', metrics=[Precision(), Recall(), c_f1], optimizer='adam')

        histroy = model.fit(x_train_val, y_train_val, validation_data=(x_test_val, y_test_val),
                            epochs=10,
                            callbacks=[rdc_lr, early_stop],
                            verbose=1, batch_size=32)

        print('val_loss: ',min(histroy.history['val_loss']))
        print('val_precision: ', max(histroy.history['val_precision']))
        print('val_recall: ', max(histroy.history['val_recall']))

        val_loss.append(min(histroy.history['val_loss']))
        val_precision.append(max(histroy.history['val_precision']))
        val_recall.append(max(histroy.history['val_recall']))

        y_score = model.predict(X_test)
        fld_score.append(y_score)
        fold_num = fold_num + 1

    return fld_score

# --- get mean score ---
# fold_result = kfold_val()
#
# test_prob = np.mean(fold_result, 0)
# y_true = np.array(y_test_mlb)
#
# print('--- Mean Precision Score --- ')
# mean_avg_precs = average_precision_score(y_true, test_prob, average='weighted')
# print(f'Weighted Mean Precision Score: {mean_avg_precs}')


############## plot history ################
# ---- visualize the history ----
def plot_history(history):
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'Precision({lr}-{batch_size}).png')
    plt.show()

    # summarize for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'Loss({lr}-{batch_size}).png')
    plt.show()

    # summarize for loss
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'Recall ({lr}-{batch_size}).png')
    plt.show()

# plot_history()







