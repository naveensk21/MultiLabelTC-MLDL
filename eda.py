import json
import matplotlib.pyplot as plt
import random
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import statistics
from sklearn.preprocessing import MultiLabelBinarizer



dataset_file_path = 'top_40_labels_dataset.json'
with open(dataset_file_path) as fp:
    dataset_json = json.load(fp)

all_label_file_path = 'label_support.json'
with open(all_label_file_path) as fp:
    all_labels = json.load(fp)

# number of mulit lables per policy text
count_num = {}
for i, datapoint in enumerate(dataset_json):
    if len(datapoint['labels']) > 2:
        count_num[i] = len(datapoint['labels'])


# plot distribution of multi labels per policy text
def plot_multi_lable_distri():
    plt.figure(figsize=(12, 6))
    x = count_num.keys()
    y = list(count_num.values())
    plt.title('Distribution of multi-labels per privacy policy')
    plt.grid()
    plt.bar(range(len(x)), y)
    plt.xlabel('Datapoints')
    plt.ylabel('No. of times a label appeared in a privacy policy')
    plt.show()


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


# plot word distribution of policy_texts
def plot_word_distri(sentence_list):
    word_length = []
    for sentence in sentence_list:
        words = sentence.split()
        senten_len = len(words)
        word_length.append(senten_len)

    plt.figure(figsize=(12, 6))
    values = word_length
    plt.title('Word distribution of the policy text')
    plt.grid()
    plt.bar(range(len(sentence_list)), values)
    plt.xlabel(f'Number of policy text: {len(sentence_list)}')
    plt.ylabel('No. of words in a policy text')
    # plt.show()

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


# plot word distribution of preprocessed policy texts
def plot_prepro_word_distri(pre_text):
    word_length = []
    for senten in pre_text:
        words = senten.split()
        senten_len = len(words)
        word_length.append(senten_len)

    print(max(word_length))
    print(min(word_length))


    plt.figure(figsize=(12, 6))
    values = word_length
    plt.title('Word distribution of the policy text')
    plt.grid()
    plt.bar(range(len(pre_text)), values)
    plt.xlabel(f'Number of policy text: {len(pre_text)}')
    plt.ylabel('No. of words in a preprocessed policy text')
    # plt.show()

plot_prepro_word_distri(preprocessed_text)
exit()
# create dictionary counters with the key as the label name and the value as the total number of labels
counters = {}
for labels in y:
    for label in labels:
        if counters.get(label) is not None:
            counters[label] += 1
        else:
            counters[label] = 1


# plot bar chart to visualize the distribution of labels (check for imbalance)
def plot_class_distribution2(count):
    plt.figure(figsize=(12, 6))
    values = list(count.values())
    name = list(count.keys())
    plt.title('All labels: Distribution of the no. of times a label appeared in a policy text')
    plt.grid()
    plt.bar(range(len(count)), values)
    plt.xlabel('Label numbers. (All 36 labels)')
    plt.ylabel('No. of times each label appeared in the dataset')
    plt.show()


def plot_all_label_distribution():
    plt.figure(figsize=(12, 6))
    x = all_labels.keys()
    y = all_labels.values()
    plt.title('Distribution of all the labels the dataset')
    plt.grid()
    plt.bar(range(len(x), y))
    plt.xlabel('All Labels')
    plt.ylabel('No. of times each label appeared in the whole dataset')
    plt.show()


