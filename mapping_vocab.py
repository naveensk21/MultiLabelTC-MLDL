from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import json
import random
import pickle
import re

# top labels file path
dataset_file_path = 'top_40_labels_dataset.json'

# gppr vocab path
gdpr_dataset_path = 'gdpr_vocab.json'

with open(gdpr_dataset_path) as fp:
    gdpr_vocab = json.load(fp)


# function to create x and y
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

# multi labels
mlb = MultiLabelBinarizer()
y_mlb = mlb.fit_transform(y)
label_classes = mlb.classes_


# load the model
loaded_model = load_model('cnn_model.h5')

# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# predict a segment text
max_len = 215
txt = X

seq = loaded_tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_len, padding='post')

def remove_tags(text):
    text = re.sub(re.compile('<.*?>'), '', text)
    text = re.sub("\s\s+", " ", text)
    return text


# tranform the predicted labels to original label string
def get_labels(predicted_labels):
    # gets all the 36 label
    mlb = [(i, label) for i, label in enumerate(label_classes)]
    # create a temporary list and sorts the probabilities in descending order with the label number.
    sort_predicted_labels = sorted([(i, pred_prob) for i, pred_prob in enumerate(list(predicted_labels))],
                                   key=lambda x: x[1], reverse=True)
    # a list with the predicted probabilities above the threshold of 0.5
    label_list = [prob for prob in sort_predicted_labels if prob[1] >= 0.5]
    # list of the the original text labels
    labels = [label_class[1] for label in label_list[:8] for label_class in mlb if label[0] == label_class[0]]

    return labels


# get the gdpr principles
def get_dgpr_vocab(predicted_label):
    vocab = []
    for label in gdpr_vocab.keys():
        if label in predicted_label:
            gdpr = gdpr_vocab.get(label)
            vocab.append(gdpr)

    return list(dict.fromkeys(vocab))


# single policy_text_predictions
def single_predicted_data(text_string):
    single_predicted_data = []

    seq = loaded_tokenizer.texts_to_sequences([text_string])
    padded_text = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = loaded_model.predict(padded_text)

    single_predicted_data.append({'policy_text': loaded_tokenizer.sequences_to_texts(seq),
                                 'predicted_labels': get_labels(pred.ravel().tolist()),
                                  'GDPR_vocab': get_dgpr_vocab(get_labels(pred.ravel().tolist()))})

    return single_predicted_data


single_data = single_predicted_data('<strong> Legal Compliance, Business Transfers and Other Disclosures. </strong> <br> <br> Notwithstanding anything to the contrary stated herein or within our Services, we may occasionally release information about users of our Services when we deem such release appropriate to comply with law, respond to compulsory process or law enforcement requests, enforce our Visitor Agreement, or protect the rights, property or safety of users of our Services, the public, Meredith Corporation, our affiliates, or any third party. Over time, we may reorganize or transfer various assets and lines of business. Notwithstanding anything to the contrary stated herein or on our Services, we reserve the right to disclose or transfer any information we collect to third parties in connection with any proposed or actual purchase, sale, lease, merger, foreclosure, liquidation, amalgamation or any other type of acquisition, disposal, transfer, conveyance or financing of all or any portion of Meredith or our affiliates. <br> <br>')

# display the associated set of labels for that policy text
print(single_data)

# p_text = single_data[0]['policy_text']
# pred_labels = single_data[0]['predicted_labels']


# multiple policy_text_predictions
def multi_predicted_data(model, text_values):
    predicted_data = []
    pred = model.predict(text_values)

    for i, datapoint in enumerate(text_values):
        predicted_data.append({'policy_text': remove_tags(X[i]),
                              'predicted_labels': get_labels(pred[i].tolist()),
                              'GDPR_vocab': []})

    return predicted_data


# predicted_data = multi_predicted_data(loaded_model, padded)
#
# print(predicted_data)


# collect the label data into a json
# with open('pred_mapped_data2.json', 'w') as fp:
#     json.dump(predicted_data, fp)





