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
gdpr_dataset_path = 'dpv_vocab.json'
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
loaded_model = load_model('model/cnn_model.h5')

# load tokenizer
with open('model/cnn_tokenizer.pickle', 'rb') as handle:
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
    # gets all the 36 label and its index
    mlb = [(i, label) for i, label in enumerate(label_classes)]
    # create a temporary list and sorts the probabilities in descending order with the label number.
    sort_predicted_labels = sorted([(i, pred_prob) for i, pred_prob in enumerate(list(predicted_labels))],
                                   key=lambda x: x[1], reverse=True)
    # a list with the predicted probabilities above the cutoff of 0.5
    pred_label_list = [prob for prob in sort_predicted_labels if prob[1] >= 0.5]
    # list of the the original text labels
    labels = [label_class[1] for label in pred_label_list for label_class in mlb if label[0] == label_class[0]]

    return labels


# get dpv vocabulary to mapped to the labels
def get_dpv_vocab(predicted_labels):
    vocab = []
    for vocabulary in gdpr_vocab.keys():
        if vocabulary in " ".join(predicted_labels):
            dpv = gdpr_vocab.get(vocabulary)
            vocab.append(dpv)
    return list(dict.fromkeys(vocab))


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


single_data = single_predicted_data("If you do not want information collected through the use of cookies, there is a simple procedure in most browsers that allows a user to accept or reject most cookies. Certain cookies that are set when some video products are accessed on our Website, called local shared objects or Flash cookies, may not be managed using the settings in your web browser. Information on managing, accepting and rejecting these cookies is available from Adobe on the Adobe website. If you set your browser or Adobe Flash options not to accept cookies or local shared objects, you may not be able to take advantage of certain Services. <br> <br>")

# display the associated set of labels single privacy policy
print(single_data)

exit()


# multiple policy_text_predictions
def multi_predicted_data(model, policy_text):
    output_data = []
    pred = model.predict(policy_text)

    for i, datapoint in enumerate(policy_text):
        output_data.append({'policy_text': remove_tags(X[i]),
                            'predicted_labels': get_pred_labels(pred[i].tolist()),
                            'DPV': get_dpv_vocab(get_pred_labels(pred[i].tolist()))})

    return output_data


predicted_data = multi_predicted_data(loaded_model, padded)

# print(predicted_data)

# collect the label data into a json
with open('pred_mapped_datatset.json', 'w') as fp:
    json.dump(predicted_data, fp)





