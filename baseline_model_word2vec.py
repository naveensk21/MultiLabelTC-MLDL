# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data manipulation
import numpy as np
import string
import json
import random
import re
import time

# Preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# For Vectorizing
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

# check the length of the sentence = sentence vector (to prevent error due to having mismatch length between the vector and sentence)
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
start = time.time()
base_classifier = BinaryRelevance(
    classifier=RandomForestClassifier(n_estimators=400, min_samples_split=2, min_samples_leaf=2, max_features='auto',
                                      max_depth=16, random_state=42),
    # classifier=LogisticRegression(C=520, tol=0.01),
    # classifier=LinearSVC(C=30, tol=0.1),
    require_dense=[True, True])

# # fit the model
base_classifier.fit(X_train_vector_avg, y_train)

end = time.time()
process = round(end-start, 2)
print(f'training time taken: {process} seconds')
# print('training time taken: ', round(time.time()-start, 0), 'seconds')

# get the predictions
y_pred = base_classifier.predict(X_test_vector_avg)

# return the models metrics - Evaluation
br_precision = precision_score(y_test, y_pred, average='macro')
br_rec = metrics.recall_score(y_test, y_pred, average='macro')
br_f1 = metrics.f1_score(y_test, y_pred, average='weighted')
br_hamm = metrics.hamming_loss(y_test, y_pred)

# display the results
print('--- Classifier Results ---')
print('Precision:', round(br_precision, 3))
print('Recall: ', round(br_rec, 3))
print('F1-score:', round(br_f1, 3))
print('Hamming Loss:', round(br_hamm, 3), '\n')

exit()
# -------- grid parameters ----------
rf_grid = {'classifier__n_estimators': [200, 300, 350, 400, 450,  500],
           'classifier__min_samples_split': [1, 2, 3],
           'classifier__min_samples_leaf': [2, 3, 4],
           'classifier__max_depth': [10, 12, 14, 16, 17, 18]
           }
lsvc_grid = {'classifier__C': [x for x in range(0, 300, 20)],'classifier__tol': [0.1, 0.01, 0.001]}
logist_grid = {'classifier__C': [x for x in range(0, 600, 20)], 'classifier__tol':[1, 0.1, 0.01, 0.001, 0.0001]}


# ------- GridSearchCV---------
def grid_search(param_grid):
    clf = GridSearchCV(base_classifier, param_grid=param_grid, scoring='f1_macro', n_jobs=-1, verbose=2)
    return clf


# ------ random search -------
def random_search(param_grid):
    rf_rgridsch = RandomizedSearchCV(estimator=base_classifier, param_distributions=param_grid, cv=5, n_iter=30)
    return rf_rgridsch


start = time.time()
search_cv = random_search(lsvc_grid)

print('Retrieving best parameters....')
search_cv.fit(X_train_vector_avg, y_train)
print(search_cv.best_params_)
print("Time take: ", round(time.time()-start, 0), 'seconds')


# best param from colab
# {'classifier__n_estimators': 400, 'classifier__min_samples_split': 2, 'classifier__min_samples_leaf': 1, 'classifier__max_features': 'auto', 'classifier__max_depth': 16}

# {'classifier__n_estimators': 400, 'classifier__min_samples_split': 2, 'classifier__min_samples_leaf': 2, 'classifier__max_depth': 16}

