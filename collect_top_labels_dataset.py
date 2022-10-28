import json
from typing import Dict

clean_dataset_file_path = 'clean_combined_data.json'

# the labels and the times they appear in the dataset
label_support_file_path = 'label_support.json'

# initialzation of x top labels
store_top_x = 40

# data with x top labels
top_labels_file_path = f'top_{store_top_x}_labels.json'

# data with x top labels and its data
top_labels_dataset_file_path = f'top_{store_top_x}_labels_dataset.json'


with open(label_support_file_path) as fp:
    label_counters: Dict[str, int] = json.load(fp)

# collects and stores the top labels (currently set at 40)
top_labels = set()
for label, counter in label_counters.items():
    if counter >= store_top_x:
        top_labels.add(label)


# dump x top labels into a json
with open(top_labels_file_path, "w") as fp:
    json.dump(list(top_labels), fp)
    print(f"Stored the labels used at least {store_top_x} times in: {top_labels_file_path}")


# load the filtered dataset with valid labels
with open(clean_dataset_file_path) as fp:
    dataset_json = json.load(fp)


# collect the dict with top labels that appears greater than equal to x times and store it in a list
print(f"Clean dataset has {len(dataset_json)} entries")
print(f"Size of labels to keep is: {len(top_labels)}")
top_x_dataset = []
for datapoint in dataset_json:
    original_labels = datapoint['labels'].copy()
    datapoint['labels'] = list({label for label in datapoint['labels'] if label in top_labels})
    how_many_labels_we_removed = len(original_labels) - len(datapoint["labels"])

    if datapoint['labels']:
        top_x_dataset.append(datapoint)

print(f"Top {store_top_x} dataset has {len(top_x_dataset)} entries")


# dump the datapoint with labels that appear more than x times into a json
with open(top_labels_dataset_file_path, "w") as f:
    json.dump(top_x_dataset, f)
    print(f"Stored the top {store_top_x} dataset in {top_labels_dataset_file_path}")
