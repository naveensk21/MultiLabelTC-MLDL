import json
from typing import Dict

clean_dataset_file_path = 'clean_combined_data.json'

# the labels and the times they appear in the dataset
label_counter_filepath = 'label_support.json'

# top 10 popular label filepath
popular_labels_filepah = 'top_10_labels.json'

# top 10 popular label and its dataset filepath
top_10_popular_label_filepath = 'top_10lb_popular_dataset.json'


# load label support
with open(label_counter_filepath) as fp:
    label_counters: Dict[str, int] = json.load(fp)

# return top 10 labels
def get_top_10_label(label_list):
    # sort the labels by occurrences
    sorted_list = sorted(label_list, key=label_list.get)
    # get the first 10 labels from the sorted list
    labels = sorted_list[-10:]

    return labels


popular_labels = get_top_10_label(label_counters)

# save the top 10 popular labels to json
with open(popular_labels_filepah, "w") as fp:
    json.dump(list(popular_labels), fp)
    print(f"Stored top 10 popular labels")


# load the filtered dataset with valid labels
with open(clean_dataset_file_path) as fp:
    dataset_json = json.load(fp)


# collect the dict with top labels that appears greater than equal to x times and store it in a list
top_x_dataset = []
for datapoint in dataset_json:
    original_labels = datapoint['labels'].copy()
    datapoint['labels'] = list({label for label in datapoint['labels'] if label in popular_labels})
    how_many_labels_we_removed = len(original_labels) - len(datapoint["labels"])

    if datapoint['labels']:
        top_x_dataset.append(datapoint)


# dump the datapoint with labels that appear more than x times into a json
with open(top_10_popular_label_filepath, "w") as f:
    json.dump(top_x_dataset, f)
    print(f"Stored the dataset in {top_10_popular_label_filepath}")








