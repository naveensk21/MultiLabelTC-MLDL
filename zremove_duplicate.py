import json
import random
import pickle
import re

dataset_file_path = 'zclean_combined_data_wcategory.json'
with open(dataset_file_path) as fp:
    dataset_json = json.load(fp)

clean_items = []
for i, datapoint in enumerate(dataset_json):
    remove_duplicate_items = list(dict.fromkeys(datapoint['labels']))
    clean_items.append(remove_duplicate_items)
    datapoint['labels'] = remove_duplicate_items

aggregated_items = []
for sublist in clean_items:
    for item in sublist:
        aggregated_items.append(item)

# total number of pretty print string with duplicates
print(len(aggregated_items))
new_joined_data = list(dict.fromkeys(aggregated_items))
# total number of pretty print string without duplicates
print(len(new_joined_data))
print(new_joined_data[:20])


with open('zzclean_dataset_wcategory.json', "w") as f:
    json.dump(dataset_json, f)


