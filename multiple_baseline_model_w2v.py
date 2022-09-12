# data manipulation
import numpy as np
import string
import json
import random
import re
import time

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import logging as logger
import tensorflow as tf

# Preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# For Vectorizing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec

# Model Selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

# Classification
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
import sklearn.metrics as metrics


# top x labels file path
dataset_file_path = 'top_40_labels_dataset.json'

# list of classfiers to test
classifiers = [
    RandomForestClassifier(n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_depth=16, max_features='auto'),
    LinearSVC(C=150, tol=0.001),
    SVC(C=20, tol=0.001),
    AdaBoostClassifier(n_estimators=200, learning_rate=0.1),
    LogisticRegression(tol=0.01, C=200),
    SGDClassifier(alpha=0.0001, max_iter=1000, tol=0.001, power_t=0.5, validation_fraction=0.1),
    KNeighborsClassifier(n_neighbors=3)
]

for classifier in classifiers:
    print(f"Classifier: {classifier.__class__}")

    def create_x_y():

        with open(dataset_file_path) as fp:
            dataset_json = json.load(fp)

        x = []  # policy texts
        y = []  # labels

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
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(re.compile('<.*?>'), '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        word_tokens = text.split()
        le = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
        word_tokens = [le.lemmatize(w) for w in word_tokens if not w in stop_words]

        cleaned_text = " ".join(word_tokens)

        return cleaned_text

    # map the function to the dataset to strip html tags
    print("Cleaning tags")
    clean_data = list(map(lambda text: preprocess_text(text), X))
    print("Cleaning tags, done")

    clean_data_tokenized = [text.split() for text in clean_data]

    # MulilabelBinarizer to vectorize the labels
    mlb = MultiLabelBinarizer()
    y_mlb = mlb.fit_transform(y)
    # print(mlb.classes_[1000:1500])

    # split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(clean_data_tokenized, y_mlb, test_size=0.3, random_state=42)

    # ---- Feature Engineering ----
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
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train])
    X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test])

    # check the length of the sentence with the sentence vector are the same
    # for i, v in enumerate(X_train_vect):
    #     print(len(X_train_tokenized[i]), len(v))

    # Compute sentence vectors by averaging the word vectors for the words contained in the sentence
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(350, dtype=float))

    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(350, dtype=float))


    # ----- Multi-label Classification with Binary Relevance  -----
    # each target variable(y1,y2,...) is treated independently and reduced to n classification problems
    start = time.time()
    wrapper_classifier = LabelPowerset(
        classifier=classifier,
        require_dense=[False, True])

    # fit the model
    wrapper_classifier.fit(X_train_vect_avg, y_train)

    end = time.time()
    process = round(end - start, 2)
    print(f'training time taken: {process} seconds')

    # get the predictions
    start = time.time()
    y_pred = wrapper_classifier.predict(X_test_vect_avg)
    print('prediction time taken....', round(time.time() - start, 0), 'seconds')

    # return the models metrics
    br_precision = precision_score(y_test, y_pred, average='macro')
    br_recall = recall_score(y_test, y_pred, average='macro')
    br_f1 = f1_score(y_test, y_pred, average='macro')
    br_hamm = hamming_loss(y_test, y_pred)
    # br_jr_score_samples = jaccard_score(y_test, y_pred, average='samples')
    # br_jr_score_macro = jaccard_score(y_test, y_pred, average='macro')

    print("Binary Relevance Precision: ", round(br_precision, 3))
    print("Binary Relevance Recall: ", round(br_recall, 3))
    print('Binary Relevance F1-score:', round(br_f1, 3))
    print('Binary Relevance Hamming Loss:', round(br_hamm, 3))
    print(' ')
    # print('Binary Relevance Jaccardi Score(samples):', round(br_jr_score_samples, 3))
    # print('Binary Relevance Jaccardi Score(samples):', round(br_jr_score_macro, 3))












