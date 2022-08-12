import json
import random
import re
import string
import time
import os
import numpy
import numpy as np
# building model
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation, Embedding, GlobalMaxPool1D, Dropout, Conv1D, LSTM, Flatten
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import KFold
# Preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
# For Vectorization
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec
from sklearn.utils import compute_class_weight
# metrics
from keras.metrics import Precision, Recall
from keras import backend as K
from sklearn.metrics import average_precision_score, precision_score
# plot
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
#optuna
import optuna
from optuna import trial
from optuna.trial import TrialState
from optuna.pruners import HyperbandPruner


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


vocab_size = len(tokenizer.word_index) + 1
encoded_ptext = tokenizer.texts_to_sequences(preprocessed_text)
maxlen = 215
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
def get_class_weights(label_classes):
    class_weights = {}
    for index, label in enumerate(label_classes):
        class_weights[index] = len(y) / counters.get(label)
    return class_weights


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


def shallow_model():
    n_labels = len(mlb.classes_)

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=maxlen, weights=[embedding_matrix]))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='linear'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(n_labels, activation='sigmoid' ))

    return model


model = shallow_model()



# --- tuning hyperparameters using optuna ---
def objective(trial):
    # clear clutter from previous sessions
    keras.backend.clear_session()

    x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train_mlb, train_size=0.8,
                                                                    test_size=0.2, random_state=42)
    # best number of hidden layers
    n_layers = trial.suggest_int('n_layers', 1, 3)

    # create trial optuna model
    opt_model = keras.Sequential()

    for i in range(n_layers):
        # best number of hidden nodes
        num_hidden_nds = trial.suggest_int(f'n_units_l{i}', 48, 240, log=True)

        # best activation function
        opt_model.add(keras.layers.Dense(num_hidden_nds,
                                         activation=trial.suggest_categorical(f'activation{i}', ['relu', 'linear'])))
        # opt_model.add(keras.layers.Dense(num_hidden_nds, input_shape=(len(ptext_pad[0]), ),
        #                                  activation=trial.suggest_categorical(f'activation{i}', ['relu', 'linear'])))

        # best dropout value
        opt_model.add(keras.layers.Dropout(rate=trial.suggest_float(f'dropout{i}', 0.0, 0.6)))

    opt_model.add(Flatten())
    # output layer
    opt_model.add(keras.layers.Dense(36, activation='sigmoid'))

    # reduce learning rate if it shows no signs of improvement
    rdc_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-05, verbose=0)

    # early stop to stop if the model stops improving
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=0, mode="auto", baseline=None,
    restore_best_weights=True)

    # compile model and get best learning rate
    learning_rate = keras.optimizers.Adam(learning_rate=trial.suggest_float('learning_rate',1e-5, 1e-3, log=True))
    opt_model.compile(loss='binary_crossentropy', metrics=[Precision(), Recall(), c_f1], optimizer=learning_rate)

    # fit the model and calculate the best batch size
    opt_histroy = opt_model.fit(x_train_cv, y_train_cv, validation_data=(x_test_cv, y_test_cv), epochs=200,
                                callbacks=[rdc_lr, early_stop],
                                verbose=0, batch_size=trial.suggest_int('size', 8, 256))

    # metrics
    val_loss = min(opt_histroy.history['val_loss'])
    val_recall = max(opt_histroy.history['val_recall'])
    val_precision = max(opt_histroy.history['val_precision'])
    val_f1 = max(opt_histroy.history['val_c_f1'])

    return val_f1


# create study to invoke optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=1500)
# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])


# display result
def multiple_metric_stats(study):
    # --- stats for multiple metrics ---
    print('--- Study statistics ---')
    # print(f'"Number of pruned trials: {len(pruned_trials)}')
    print(f"Number of completed trials: {len(study.trials)}")

    for study_trials in study.best_trials:
        metrics = study_trials.values
        parameters = study_trials.params
        trial_num = study_trials.number
        print(f'Trial Number: {trial_num}')
        print(f'Metric: {metrics} ')
        print('Params')
        for key, value in parameters.items():
            print(f'   {key}: {value}')
        print(" ")


def single_metric_stats(study):
    # --- stats for single metrics ---
    print(f"Number of completed trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best trial value: {study.best_trial.value}")
    print("  Best Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


def optuna_visual(study):
    # plot the trials from the study
    optuna_fig = optuna.visualization.plot_optimization_history(study)
    optuna_fig.show()


# multiple_metric_stats(study)
single_metric_stats(study)
# optuna_visual(study)


# k fold validation
def kfold_val():
    cross_val = KFold(n_splits=5, shuffle=True, random_state=42)
    val_loss = []
    eval_score = []
    lr = 0.01
    opt = keras.optimizers.Adam(learning_rate=lr)
    fold_num = 0
    for train_i, test_i in cross_val.split(X_train, y_train_mlb):
        keras.backend.clear_session()

        # generate print
        print('------------------------------------------')
        print(f'Training for fold {fold_num}....')
        print('------------------------------------------')

        x_train_val, x_test_val = X_train[train_i], X_train[test_i]
        y_train_val, y_test_val = y_train_mlb[train_i], y_train_mlb[test_i]

        model = shallow_model()

        rdc_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-06, verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto",
                                   baseline=None,
                                   restore_best_weights=True)

        model.compile(loss='binary_crossentropy', metrics=[Precision(), Recall(), c_f1], optimizer='adam')

        histroy = model.fit(x_train_val, y_train_val, validation_data=(x_test_val, y_test_val),
                            epochs=10,
                            callbacks=[rdc_lr,early_stop],
                            verbose=1, batch_size=32)

        print(min(histroy.history['val_loss']))
        val_loss.append(min(histroy.history['val_loss']))
        y_score = model.predict(X_test)
        eval_score.append(y_score)
        fold_num = fold_num + 1

    return eval_score


# eval_score = kfold_val()
#
# test_prob = np.mean(eval_score, 0)
# y_true = np.array(y_test_mlb)
#
# print('--- Mean Precision Score --- ')
# mean_avg_precs = average_precision_score(y_true, test_prob, average='weighted')
# print(f'Weighted Mean Precision Score: {mean_avg_precs}')


# ---- visualize the history ----
# print(history.history.keys())
# summarize for precision
# def plot_history():
#     plt.plot(history.history['precision'])
#     plt.plot(history.history['val_precision'])
#     plt.title('Model Precision')
#     plt.ylabel('precision')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig(f'figures/Precision({lr}-{batch_sizes}).png')
#     plt.show()
#
#     # summarize for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Loss')
#     plt.ylabel('precision')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig(f'figures/Loss({lr}-{batch_sizes}).png')
#     plt.show()
#
#     # summarize for loss
#     plt.plot(history.history['recall'])
#     plt.plot(history.history['val_recall'])
#     plt.title('Model recall')
#     plt.ylabel('recall')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig(f'figures/Recall ({lr}-{batch_sizes}).png')
#     plt.show()











