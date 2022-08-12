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
    MultinomialNB(),
    # NearestNeighbors(n_neighbors=5, radius=1.0, leaf_size=30),
    RandomForestClassifier(n_estimators=100),
    LinearSVC(C=400, tol=0.1),
    SVC(),
    AdaBoostClassifier(),
    LogisticRegression()
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

    print('training time taken: ', round(time.time() - start, 0), 'seconds')

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

    # # parameter grid
    # grid_param = {
    #     'classifier': [wrapper_classifier],
    #     'classifier__classifier': [classifier],
    #     # 'classifier__C': [800, 700, 600, 400, 200, 100, 60, 50, 40, 30],
    #     'classifier__classifier__n_estimators': [100, 200, 300, 400, 500],
    # }
    # # grid search
    # clf = GridSearchCV(wrapper_classifier, param_grid=grid_param, scoring='f1_macro')
    #
    # # print(clf.estimator.get_params().keys())
    #
    # # fit to estimate the parameters
    # clf.fit(X_train_tfidf, y_train)
    #
    # # displat the best parameters
    # print(clf.best_params_)











