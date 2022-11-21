from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import json
import random
import pickle
import re

# top labels file path
dataset_file_path = 'zzclean_dataset.json'
with open(dataset_file_path) as fp:
    dataset_json = json.load(fp)

# gppr vocab path
gdpr_dataset_path = 'gdpr_vocab.json'
with open(gdpr_dataset_path) as fp:
    gdpr_vocab = json.load(fp)


# function to create x and y
def create_x_y():

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

# multi labels
mlb = MultiLabelBinarizer()
y_mlb = mlb.fit_transform(y)
label_classes = mlb.classes_
print(label_classes)

# load the model
loaded_model = load_model('zcnn_model.h5')

# load tokenizer
with open('ztokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# predict a segment text
max_len = 215
txt = X

seq = loaded_tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len, padding='post')

# print(X[5])
# pred = loaded_model.predict(padded)
# print(pred[5].astype(str))
# exit()


def remove_tags(text):
    text = re.sub(re.compile('<.*?>'), '', text)
    text = re.sub("\s\s+", " ", text)
    return text


# transform the predicted labels to original label string
def get_pred_labels(predicted_labels):
    # gets all the 36 label
    mlb = [(i, label) for i, label in enumerate(label_classes)]
    # create a temporary list and sorts the probabilities in descending order with the label number.
    sort_predicted_labels = sorted([(i, pred_prob) for i, pred_prob in enumerate(list(predicted_labels))],
                                   key=lambda x: x[1], reverse=True)
    # a list with the predicted probabilities above the cutoff of 0.5
    label_list = [prob for prob in sort_predicted_labels if prob[1] >= 0.5]
    # list of the the original text labels
    labels = [label_class[1] for label in label_list[:8] for label_class in mlb if label[0] == label_class[0]]

    return labels


# get dpv vocabulary to mapped to the labels
def get_dpv_vocab(labels):
    vocab = []
    for vocabulary in gdpr_vocab.keys():
        if vocabulary in " ".join(labels):
            dpv = gdpr_vocab.get(vocabulary)
            vocab.append(dpv)

    clean_vocab = list(dict.fromkeys(vocab))
    return clean_vocab


def get_labels(text):
    for datapoint in dataset_json:
        if text == datapoint['policy_text']:
            return datapoint['labels']


# single policy_text_predictions
def single_predicted_data(text_string):
    output_data = []

    seq = loaded_tokenizer.texts_to_sequences([text_string])
    padded_text = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = loaded_model.predict(padded_text)

    output_data.append({'policy_text': loaded_tokenizer.sequences_to_texts(seq),
                        'predicted_data_practice': get_pred_labels(pred.ravel().tolist()),
                        'fine_grained_annotations': get_labels(text_string),
                        'DPV': get_dpv_vocab(get_labels(text_string))})

    return output_data


single_data = single_predicted_data("<strong> How to Correct or Update Your Information: </strong> <br> <br> Meredith Corporation provides you with the ability to access and edit certain personally identifying information that you have provided to us through our Services. To update this information, please visit the \"My Account\" area or comparable feature of the Service you used to enter your information. If you cannot locate such a feature, send us an email at privacy@meredith.com. <br> <br>")

# display the associated set of labels for that policy text
# print(single_data)


# multiple policy_text_predictions
def multi_predicted_data(model, text_values):
    output_data = []
    pred = model.predict(text_values)

    for i, datapoint in enumerate(text_values):
        output_data.append({'policy_text': remove_tags(X[i]),
                            'predicted_label': get_pred_labels(pred[i].tolist()),
                            'label_string': get_labels(X[i]),
                            'DPV': get_dpv_vocab(get_labels(X[i]))})

    return output_data


predicted_data = multi_predicted_data(loaded_model, padded)

# collect the label data into a json
with open('zpred_mapped_data.json', 'w') as fp:
    json.dump(predicted_data, fp)





