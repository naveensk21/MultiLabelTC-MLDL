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
import pandas as pd
import seaborn as sns

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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain
from sklearn.ensemble import RandomForestClassifier


# Evaluation
import sklearn.metrics as metrics
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

# top 10 popular labels -> 630 data
# top 36 labels         -> 1186 data
# all labels (post-cl) -> 3065 data
# all label (pre-cl)    -> 3792 data

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

    print(len(dataset_json))
    # print(f"Loaded {len(x)} policies with {len(y)} corresponding sets of policy practices")
    return x, y


# create x and y
X, y = create_x_y()

# ----- Preprocessing -----
# function to preprocess the policy text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(re.compile('<.*?>'), '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('\d+', '', text)
    text = re.sub(r"\bhttp\w+", "", text) # remove words that start with http

    word_tokens = text.split()
    le=WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    word_tokens = [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    cleaned_text = " ".join(word_tokens)
    cleaned_text = re.sub(r"\b[a-zA-Z]\b", "", cleaned_text)
    cleaned_text = " ".join(cleaned_text.split())

    return cleaned_text


# map the function to preprocess the data
clean_data = list(map(lambda text: preprocess_text(text), X))

print(f"Original Text: {X[1]}")
print(f"Cleaned Text: {clean_data[1]}")
exit()


# Use MultilabelBinarizer to vectorize the labels
mlb = MultiLabelBinarizer()
y_mlb = mlb.fit_transform(y)
n_classes = y_mlb.shape[1]

# print(f"Labels: {y[1]}")
# print(f"Encoded Labels: {y_mlb[1]}")

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
    classifier=RandomForestClassifier(n_estimators=200,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_features='auto',
                                      max_depth=50, random_state=42),
    # classifier=KNeighborsClassifier(n_neighbors=3, weights='distance', leaf_size=40, metric='minkowski'),
    # classifier=LinearSVC(C=16, tol=0.01, dual=False),
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

# return the models metrics
br_prec = metrics.precision_score(y_test, y_pred, average='micro')
br_rec = metrics.recall_score(y_test, y_pred, average='micro')
br_f1 = metrics.f1_score(y_test, y_pred, average='micro')
br_hamm = metrics.hamming_loss(y_test, y_pred)

# display score
print('Precision: ', round(br_prec, 3))
print('Recall: ', round(br_rec, 3))
print('F1-score:', round(br_f1, 3))
print('Hamming Loss:', round(br_hamm, 3))

exit()
# classification report and confusion matrixs for each label
print(classification_report(y_test, y_pred, target_names=[f'label-{i}' for i, label in enumerate(mlb.classes_)]))
confusion = multilabel_confusion_matrix(y_test, y_pred)

# ----------------- gridsearch -----------------
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
    'classifier__C': [x for x in range(200)]
}


def grid_search(params):
    clf = GridSearchCV(base_classifier, param_grid=params, cv=3,  scoring='f1_micro')
    return clf


def random_search(params):
    clf = RandomizedSearchCV(estimator=base_classifier, param_distributions=params, cv=5, n_iter=50, n_jobs=-1)
    return clf


clf = grid_search(grid_param_svm)
# clf = random_search(grid_param_svm)
clf.fit(X_train_tfidf, y_train)
print('Retrieving best parameters....')
print(clf.best_params_)

exit()


# confusion matrix definition
vis_array = confusion
labels = ["".join("label " + str(i)) for i in range(0, 36)]

# ref: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
# https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=8):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label', fontsize=fontsize)
    axes.set_xlabel('Predicted label', fontsize=fontsize)
    axes.set_title(class_label, fontsize=fontsize)


fig, ax = plt.subplots(6, 6, figsize=(12, 7))

for axes, cfs_matrix, label in zip(ax.flatten(), vis_array, labels):
    print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

fig.tight_layout()
# # fig.savefig("image2.png")
# plt.show()


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


# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test_tfidf, y_test)
# print(result)






