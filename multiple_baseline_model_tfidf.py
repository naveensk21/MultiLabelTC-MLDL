import json
import random
import re
import time
import numpy as np

# model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

# preprocessing
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, jaccard_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold

# warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# top x labels file path
dataset_file_path = 'top_20_labels_dataset.json'

# list of classfiers to test
classifiers = [
    RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=1, max_depth=50, max_features='auto'),
    LinearSVC(C=150, tol=0.001),
    SVC(C=20, tol=0.003),
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
    def strip_html_tags(text):
        text = text.lower() # lower the characters
        text = re.sub(re.compile('<.*?>'), '', text) # strip html tags
        return text

    # map the function to the dataset to strip html tags
    print("Cleaning tags")
    clean_data = list(map(lambda text: strip_html_tags(text), X))
    print("Cleaning tags, done")

    # MulilabelBinarizer to vectorize the labels
    mlb = MultiLabelBinarizer()
    y_mlb = mlb.fit_transform(y)
    # print(mlb.classes_[1000:1500])

    # split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(clean_data, y_mlb, test_size=0.3, random_state=42)

    # ---- Feature Engineering ----
    # tfidf-vectorize the policy text
    tfidf = TfidfVectorizer(max_features=10000, stop_words=stop_words)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # print(tfidf.vocabulary_)

    # shapes of x and y
    # print(X_train_tfidf.shape)
    # print(X_test_tfidf.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    # ----- Multi-label Classification with Binary Relevance  -----
    # each target variable(y1,y2,...) is treated independently and reduced to n classification problems
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    start = time.time()
    wrapper_classifier = BinaryRelevance(
        classifier=classifier,
        require_dense=[False, True])

    # fit the model
    wrapper_classifier.fit(X_train_tfidf, y_train)

    end = time.time()
    process = round(end - start, 2)
    print(f'training time taken: {process} seconds')

    # get the predictions
    start = time.time()
    y_pred = wrapper_classifier.predict(X_test_tfidf)
    print('prediction time taken....', round(time.time() - start, 0), 'seconds')

    # return the models metrics
    br_precision = precision_score(y_test, y_pred, average='macro')
    br_recall = recall_score(y_test, y_pred, average='macro')
    br_f1 = f1_score(y_test, y_pred, average='weighted')
    br_hamm = hamming_loss(y_test, y_pred)
    # br_jr_score_samples = jaccard_score(y_test, y_pred, average='samples')
    # br_jr_score_macro = jaccard_score(y_test, y_pred, average='macro')

    print("Binary Relevance Precision: ", round(br_precision, 3))
    print("Binary Relevance Recall: ", round(br_recall, 3))
    print('Binary Relevance F1-score:', round(br_f1, 3))
    print('Binary Relevance Hamming Loss:', round(br_hamm, 3))
    # print('Binary Relevance Jaccardi Score(samples):', round(br_jr_score_samples, 3))
    # print('Binary Relevance Jaccardi Score(samples):', round(br_jr_score_macro, 3))












