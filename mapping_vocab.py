from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import json
import random
import pickle
import re

# top labels file path
dataset_file_path = 'top_40_labels_dataset.json'
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
def get_dpv_vocab(predicted_labels):
    vocab = []
    for vocabulary in gdpr_vocab.keys():
        if vocabulary in " ".join(predicted_labels):
            dpv = gdpr_vocab.get(vocabulary)
            vocab.append(dpv)
    return vocab


# function to get the original labels
def get_orig_labels(policy_text):
    labels = []
    for datapoint in dataset_json:
        if policy_text in datapoint['policy_text']:
            label = datapoint['labels']
            labels.append(label)
    return labels


# single policy_text_predictions
def single_predicted_data(text_string):
    output_data = []

    seq = loaded_tokenizer.texts_to_sequences([text_string])
    padded_text = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = loaded_model.predict(padded_text)

    output_data.append({'policy_text': loaded_tokenizer.sequences_to_texts(seq),
                        'predicted_labels': get_pred_labels(pred.ravel().tolist()),
                        'DPV': get_dpv_vocab(get_pred_labels(pred.ravel().tolist()))})

    return output_data


single_data = single_predicted_data("<strong> Email. </strong> <br> <br> You can opt-out from any Meredith email newsletter or commercial email list and prevent the collection of related email response information by our email service providers by using the unsubscribe link at the bottom of each message and/ or by visiting the Email Preferences page on our sites and updating your preferences. If you no longer want to receive third-party email offers that you requested through our Services, simply follow the advertiser's unsubscribe link or opt-out instructions that should be included in every commercial message you receive. <br> <br>")

# display the associated set of labels for that policy text
print(single_data)


# multiple policy_text_predictions
def multi_predicted_data(model, text_values):
    output_data = []
    pred = model.predict(text_values)

    for i, datapoint in enumerate(text_values):
        output_data.append({'policy_text': remove_tags(X[i]),
                              'predicted_labels': get_pred_labels(pred[i].tolist()),
                              'DPV': get_dpv_vocab(get_pred_labels(pred[i].tolist()))})

    return output_data


predicted_data = multi_predicted_data(loaded_model, padded)
#
# print(predicted_data)


# collect the label data into a json
with open('pred_mapped_data2.json', 'w') as fp:
    json.dump(predicted_data, fp)





