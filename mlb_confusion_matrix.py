import json
import random
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from sklearn.metrics import roc_curve, auc
from itertools import cycle

# top labels file path
dataset_file_path = 'top_40_labels_dataset.json'
with open(dataset_file_path) as fp:
    dataset_json = json.load(fp)

# function to create x and y
def create_x_y():

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

# multi labels
mlb = MultiLabelBinarizer()
y_mlb = mlb.fit_transform(y)
label_classes = mlb.classes_

# load the model
loaded_model = load_model('model/hybrid_model.h5')
loaded_model2 = load_model('model/cnn_model.h5')

# load tokenizer
with open('model/hybrid_tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

with open('model/hybrid_tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer2 = pickle.load(handle)

# predict a segment text
max_len = 215
txt = X

seq = loaded_tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(padded, y_mlb, test_size=0.2, random_state=42)

preds = loaded_model.predict(X_test)
preds = np.where(preds < 0.5, 0, 1)

# Creating multilabel confusion matrix
confusion = multilabel_confusion_matrix(y_test, preds)

vis_array = confusion
print(confusion)
# for matrix in confusion:
#     print(matrix[0][0])


def plot_multi_roc(models):
    plt.figure(0).clf()
    n_classes = len(mlb.classes_)
    # def get_tokenizer():
    #     for tokeniz in tokenizer:
    #         return tokeniz

    for model in models:
        y_score = model.predict(X_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2  # line_width
        plt.plot(fpr[3], tpr[3], color='darkorange',
                 lw=lw, label=f'ROC curve {type(model).__name__} (area = %0.2f)' % roc_auc[3])  # Drawing Curve according to 3. class
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


# ref -> https://scikit-learn.org/1.1/auto_examples/model_selection/plot_roc.html
# plot roc for the loaded model
def plot_roc():
    y_score = loaded_model.predict(X_test)
    # Learn to predict each class against the other

    n_classes = len(mlb.classes_) # number of class
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2 # line_width
    plt.plot(fpr[3], tpr[3], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[3]) # Drawing Curve according to 3. class
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()



    # Process of plotting roc-auc curve belonging to all classes.
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "lightcoral", "maroon", "peachpuff", "papayawhip", "yellow", "darkolivegreen", "honeydew", "limegreen", "lightseagreen", "mediumspringgreen", "moccasin", "cadetblue", "lightpink", "blueviolet", "tomato","dimgrey","teal","slateblue", "azure","orchid","navy","lemonchiffon","hotpink","thistle","lavender","cornflowerblue","chartreuse","chocolate","indianred","darkcyan","mediumorchid","rebeccapurple","azure"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for each target label")
    # plt.legend(loc="lower right")
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "lightcoral", "maroon", "peachpuff", "papayawhip", "yellow", "darkolivegreen", "honeydew", "limegreen", "lightseagreen", "mediumspringgreen", "moccasin", "cadetblue", "lightpink", "blueviolet", "tomato","dimgrey","teal","slateblue", "azure","orchid","navy","lemonchiffon","hotpink","thistle","lavender","cornflowerblue","chartreuse","chocolate","indianred","darkcyan","mediumorchid","rebeccapurple","azure"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    plt.show()


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
# fig.savefig("image2.png")
plt.show()


