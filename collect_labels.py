import os
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


# gets all the files as a list from the path
def get_files_from_path(path: str) -> list:
    all_files = glob.glob(path + '/*.csv')
    return all_files


# load a single csv file with the policy_name, segment, label
def load_csv_file(filename: str) -> list:
    data = []
    file_path = f'dataset/pretty_print/{filename}'
    filename = os.path.basename(file_path).split('.')[0]

    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append({'policy_name': filename,
                         'segment_id': int(row[1]),
                         'label': row[3]})
    return data


# gets all the labels per segment in a file
def get_labels_per_file(filename: str):
    list_data = load_csv_file(filename)
    labels_dict = defaultdict(list)

    for element in list_data:
        segment_id = element['segment_id']
        labels = element['label']

        labels_dict[segment_id].append(labels)

    return labels_dict


# removes the duplicated segments in dataset
def remove_dupl_segme(filename: str):
    seen = set()
    new_data_list = []
    csv_file = load_csv_file(filename)
    for d in csv_file:
        t = tuple(d.items())
        if t[1] not in seen:
            seen.add(t[1])
            new_data_list.append(d)
    new_data_list.sort(key=lambda elem: elem['segment_id'])
    return new_data_list

# label dict - contains the dictionary with all the labels associated with a particular segment
# array_list - contains an array list with policy_name, segment, and label
# label_dict = get_labels_per_file('post-gazette.com.csv')
# gaz_array = load_csv_file('post-gazette.com.csv')
# gaz_array_dupli = remove_dupl_segme('post-gazette.com.csv')


# assign multiple labels to a segment in the file
def assigning_labels_to_seg(filename: str):
    dictionary_labels = get_labels_per_file(filename)
    file_array_list = remove_dupl_segme(filename)

    data = []
    count = 0-1
    for line in file_array_list:
        segment_id = line['segment_id']
        policy_name = line['policy_name']

        if segment_id in sorted(dictionary_labels.keys()):
            count = count + 1
            data.append({'policy_name': policy_name,
                      'segment_id': segment_id,
                      'labels': dictionary_labels.get(count)})
    return data


# Load all the csv files and appends the segment_id, and labels to each policy_name
def load_all_csv_files(path: str):
    data = []

    all_files = get_files_from_path(path)

    for file in all_files:
        file_basename = os.path.basename(file)
        data.append(assigning_labels_to_seg(file_basename))

    return data


collect_labels = load_all_csv_files('dataset/pretty_print')

# collect the label data into a json
with open('collected_labels.json', 'w') as fp:
    json.dump(collect_labels, fp)



# ===============================end-testing====================================




















