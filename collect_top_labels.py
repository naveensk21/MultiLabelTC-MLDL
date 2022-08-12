import json
from typing import Dict

label_support_file_path = 'label_support.json'
store_top_x = 40

# data with the x top labels
top_labels_file_path = f'top_{store_top_x}_labels.json'


with open(label_support_file_path) as fp:
    label_counters: Dict[str, int] = json.load(fp)

top_labels = set()
for label, counter in label_counters.items():
    if counter >= store_top_x:
        top_labels.add(label)


with open(top_labels_file_path, "w") as fp:
    json.dump(list(top_labels), fp)
    print(f"Stored the labels used at least {store_top_x} times in: {top_labels_file_path}")

