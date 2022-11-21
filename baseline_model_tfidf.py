import json
import random
import re # regex
import time
import pickle
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

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
from sklearn.model_selection import train_test_split

# Classification
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.neighbors import KNeighborsClassifier


# Evaluation
import sklearn.metrics as metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay


dataset_file_path = 'zclean_combined_data_wcategory.json'


def create_x_y():

    with open(dataset_file_path) as fp:
        dataset_json = json.load(fp)

    x = [] # policy texts
    y = [] # labels

    random.shuffle(dataset_json)
    for datapoint in dataset_json:
        x.append(datapoint['policy_text'])
        y.append(datapoint['data_practice'])

    # print(f"Loaded {len(x)} policies with {len(y)} corresponding sets of policy practices")
    return x, y


# create x and y
X, y = create_x_y()


# ----- Preprocessing -----
# function to preprocess the policy text
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


# map the function to preprocess the data
clean_data = list(map(lambda text: preprocess_texts(text), X))

# Use MultilabelBinarizer to vectorize the labels

mlb = MultiLabelBinarizer()
y_mlb = mlb.fit_transform(y)
n_classes = y_mlb.shape[1]

# print(mlb.classes_[1000:1500])
# print(y_mlb[324])
# print(y_mlb[14])


# split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(clean_data, y_mlb, test_size=0.2, random_state=42)

# ---- Feature Engineering ----
# tfidf-vectorizer the policy text
tfidf = TfidfVectorizer(analyzer='word')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# shapes of x and y
# print(X_train_tfidf.shape)
# print(X_test_tfidf.shape)
# print(y_train.shape)
# print(y_test.shape)

# ----- Binary Relevance with RandomForestClassifier -----
# each target variable(y1,y2,...) is treated independently and reduced to n classification problems
start=time.time()
base_classifier = BinaryRelevance(
    # classifier=RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=1, max_features='auto',
    #                                   max_depth=50, random_state=42),
    # classifier=LogisticRegression(C=200, tol=0.01),
    classifier=LinearSVC(C=13, tol=0.1, dual=False),

    require_dense=[False, True])

# fit the model
base_classifier.fit(X_train_tfidf, y_train)
# {'classifier__tol': 0.001, 'classifier__C': 167}

# save model
# filename = 'ml_model.sav'
# pickle.dump(base_classifier, open(filename, 'wb'))

end = time.time()
process = round(end-start, 2)
print(f'training time taken: {process} seconds')

# get the predictions
y_pred = base_classifier.predict(X_test_tfidf)

# classification report for each label
# print(classification_report(y_test, y_pred, target_names=[f'label-{i}' for i, label in enumerate(mlb.classes_)]))

# return the models metrics
br_prec = metrics.precision_score(y_test, y_pred, average='micro')
br_rec = metrics.recall_score(y_test, y_pred, average='micro')
br_f1 = metrics.f1_score(y_test, y_pred, average='micro')
br_hamm = metrics.hamming_loss(y_test, y_pred)
# clf_rep = multilabel_confusion_matrix(y_test, y_pred)

# display score
print('Precision: ', round(br_prec, 3))
print('Recall: ', round(br_rec, 3))
print('F1-score:', round(br_f1, 3))
print('Hamming Loss:', round(br_hamm, 3))


# ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
def plot_pr_curve():
    classifier = OneVsRestClassifier(LogisticRegression(tol=0.01, C=200))
    classifier.fit(X_train_tfidf, y_train)
    y_score = classifier.decision_function(X_test_tfidf)
    y_predict = classifier.predict(X_test_tfidf)

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot()
    _ = display.ax_.set_title("micro-averaged over all classes")

    plt.show()


# ----- gridsearch ------
# grid_param_lsvc = {
#     'classifier': [BinaryRelevance()],
#     'classifier__classifier': [LinearSVC()],
#     'classifier__classifier__C':[1.0, 5.0, 10.0, 15.0, 20.0],
#     'classifier_classifier__tol': [0.0001, 0.00001, 0.000001, 0.00000001]
# }
grid_param_rf = {
    'classifier__n_estimators': [200, 300, 400, 500, 600],
    'classifier__min_samples_split': [2, 4, 6],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_depth': [16, 18, 20, 30, 40, 50]
    }

grid_param_knn = {
    'classifier__n_neighbors': [3,5,7,9,11,13],
    'classifier__leaf_size' : [20, 30, 40, 50],
    'classifier__weights' : ['uniform', 'distance'],
    'classifier__metric' : ['minkowski', 'euclidean', 'manhattan']
}

grid_param_lr = {
    'classifier__tol': [0.1, 0.01, 0.001, 0.0001],
    'classifier__C': [x for x in range(400)]
}

grid_param_svm = {
    'classifier__tol': [0.1, 0.01, 0.001, 0.0001],
    'classifier__C': [x for x in range(500)]
}


def grid_search(params):
    clf = GridSearchCV(base_classifier, param_grid=params, cv=3,  scoring='f1_micro')
    return clf


def random_search(params):
    clf = RandomizedSearchCV(estimator=base_classifier, param_distributions=params, cv=5, n_iter=50, n_jobs=-1)
    return clf


# clf = grid_search(grid_param_svm)
clf = random_search(grid_param_svm)
clf.fit(X_train_tfidf, y_train)
print('Retrieving best parameters....')
print(clf.best_params_)


# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test_tfidf, y_test)
# print(result)



# classifier=LinearSVC(C=386, tol=0.1, dual=False),
# Precision:  0.736
# Recall:  0.486
# F1-score: 0.585
# Hamming Loss: 0.028

# classifier=LinearSVC(C=98, tol=0.1, dual=False),
# Precision:  0.725
# Recall:  0.503
# F1-score: 0.594


