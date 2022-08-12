import json
import random
import re # regex
import time
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import logging as logger
import tensorflow as tf
from keras import backend as K

# data manipulation
import numpy as np
import string

# Preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# For Vectorizing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec

# Model Selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score

# Classification
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay


dataset_file_path = 'top_40_labels_dataset.json'

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
# function to strip the html tags
def preprocess_texts(text):
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
print('preprocessing data...')
clean_data = list(map(lambda text: preprocess_texts(text), X))
print('preprocessing done...')

# Use MulilabelBinarizer to vectorize the labels
mlb = MultiLabelBinarizer()
y_mlb = mlb.fit_transform(y)
# print(mlb.classes_[1000:1500])

# tokenize the data
clean_data_tokenized = [text.split() for text in clean_data]

# split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(clean_data_tokenized, y_mlb, test_size=0.3, random_state=42)

# -----Word2Vec -----
# create a Word2vec model
w2v_model = Word2Vec(X_train, size=350, window=10, min_count=1)

# train the model
w2v_model.train(X_train, epochs=10, total_examples=len(X_train))

# total number of extracted words
vocab = w2v_model.wv.vocab
print("The total number of words are : ", len(vocab))

# Generate aggregated sentence vectors based on the word vectors for each word in the sentence
words = set(vocab)
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in lst if i in words])for lst in X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in lst if i in words]) for lst in X_test])

# check the length of the sentence with the sentence vector are the same
# for i, v in enumerate(X_train_vect):
#     print(len(X_train_tokenized[i]), len(v))

# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
X_train_vector_avg = []
for vector in X_train_vect:
    if vector.size:
        X_train_vector_avg.append(vector.mean(axis=0))
    else:
        X_train_vector_avg.append(np.zeros(100, dtype=float))

X_test_vector_avg = []
for vector in X_test_vect:
    if vector.size:
        X_test_vector_avg.append(vector.mean(axis=0))
    else:
        X_test_vector_avg.append(np.zeros(100, dtype=float))

# check sentence vector lengths are consistent
# for i, v in enumerate(X_train_vect_avg):
#     print(len(X_train[i]), len(v))


# ----- Binary Relevance with RandomForestClassifier -----
# each target variable(y1,y2,...) is treated independently and reduced to n classification problems
np.random.seed(42)
start=time.time()
base_classifier = BinaryRelevance(
    classifier=RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_features='auto',
                                      max_depth=16, random_state=42),
    # classifier=LinearSVC(C=30, tol=0.1),
    require_dense=[False, True])

# # fit the model
base_classifier.fit(X_train_vector_avg, y_train)

print('training time taken: ', round(time.time()-start, 0), 'seconds')

# get the predictions
start=time.time()
y_pred = base_classifier.predict(X_test_vector_avg)

print('prediction time taken....', round(time.time()-start, 0), 'seconds \n')

# return the models metrics - Evaluation
br_precision = precision_score(y_test, y_pred, average='macro')
br_rec = metrics.recall_score(y_test, y_pred, average='macro')
br_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
br_hamm = metrics.hamming_loss(y_test, y_pred)

# display the results
print('Classifier Results')
print('Prediction Score:', round(br_precision, 3))
print('Recall: ', round(br_rec, 3))
print('F1-score:', round(br_f1, 3))
print('Hamming Loss:', round(br_hamm, 3))

"""
Classifier Results
Prediction Score: 0.461
Recall:  0.254
F1-score: 0.361
Hamming Loss: 0.032
"""

# ------- GridSearchCV---------
# grid_param = {
#     'classifier__n_estimators': [300, 400, 500],
#     'classifier__min_samples_split': [1, 2, 3],
#     'classifier__min_samples_leaf': [1, 2, 3],
#     }
#
# clf = GridSearchCV(base_classifier, param_grid=grid_param, scoring='f1_macro', n_jobs=-1, verbose=2)
# print('Getting Best Parameter----')
# start=time.time()
# clf.fit(X_train_vect_avg, y_train)
#
# # print(clf.estimator.get_params().keys())
# print(clf.best_params_)
# print(f'Time taken to run grid search: ', round(time.time()-start, 0), 'seconds')

# ------ random search -------
# rf_grid = {'classifier__n_estimators': [300, 400],
#            'classifier__min_samples_split': [1,2],
#            'classifier__min_samples_leaf': [2, 3],
#            'classifier__max_depth': [16, 17],
#            }
#
# lsvc_grid = {'classifier__C': [10, 20, 30, 40, 50, 60, 100, 200],'classifier__tol': [0.1, 0.01, 0.001]}
#
# # logist_grid = {'classifier__C': [20, 50, 80, 100, 200, 300, 400, 500], 'classifier__tol':[1, 0.1, 0.01, 0.001, 0.0001]}
#
# rf_rgridsch = RandomizedSearchCV(estimator=base_classifier, param_distributions=lsvc_grid, cv=5, n_iter=80)
# start = time.time()
# print('Getting Best Parameter----')
# rf_rgridsch.fit(X_train_vect_avg, y_train)
#
# print(rf_rgridsch.best_params_)
# print("Time take: ", round(time.time()-start, 0), 'seconds')

#best param from colab
# {'classifier__n_estimators': 400, 'classifier__min_samples_split': 2, 'classifier__min_samples_leaf': 1, 'classifier__max_features': 'auto', 'classifier__max_depth': 16}


#optuna
# import optuna
# from optuna import trial
#
# def objective(trial):
#
#     n_estimators = trial.suggest_categorical('n_estimators', [200, 300, 400, 500, 600])
#     min_samples_split = trial.suggest_categorical('min_sample_split', [1, 2, 3])
#     min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [2, 3, 4])
#     max_depth = trial.suggest_categorical('max_depths', [15, 16, 17, 18])
#
#     br_rf = BinaryRelevance(
#         classifier=RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
#                                           min_samples_leaf=min_samples_leaf,
#                                           max_features='auto',
#                                           max_depth=max_depth,
#                                           random_state=42),
#         # classifier=LinearSVC(C=30, tol=0.1),
#         require_dense=[False, True])
#
#     score = cross_val_score(br_rf, X_train, y_train, cv=5, scoring="f1")
#     rf_f1_mean = score.mean()
#
#     return rf_f1_mean
#
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50, timeout=None)
#
# optimized_rf = BinaryRelevance(RandomForestClassifier(n_estimators=study.best_params['n_estimators'],
#                                                       min_samples_split=study.best_params['min_samples_split'],
#                                                       min_samples_leaf=study.best_params['min_samples_leaf'],
#                                                       max_features='auto',
#                                                       max_depth=study.best_params['max_depth'],
#                                                       random_state=42))
#
# optimized_rf.fit(X_train_vector_avg, y_train)
