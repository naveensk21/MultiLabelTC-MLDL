import os
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path


def get_files_from_path(path: str) -> list:
    all_files = glob.glob(path + '/*.csv')
    return all_files


def load_csv_file(filename: str) -> list:
    data = []
    file_path = f'dataset/annotations/{filename}'
    filename = os.path.basename(file_path).split('.')[0]

    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append({'policy_name': filename,
                         'segment_id': int(row[4]),
                         'data_practice_category': row[5]})
    return data


# get the data practice category per file
def get_category_per_file(filename: str):
    list_data = load_csv_file(filename)
    cate_dict = defaultdict(list)
    new_list = dict()

    for element in list_data:
        segment_id = element['segment_id']
        category = element['data_practice_category']

        cate_dict[segment_id].append(category)

    # remove duplicate categories
    for i, cate in cate_dict.items():
        clean_cate = list(dict.fromkeys(cate))
        new_list[i] = (clean_cate)

    return new_list


# remove duplicate segments
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

# assign the data pratice category to the segments
def assigning_category_to_seg(filename: str):
    dictionary_category = get_category_per_file(filename)
    file_array_list = remove_dupl_segme(filename)

    data = []
    count = 0-1
    for line in file_array_list:
        segment_id = line['segment_id']
        policy_name = line['policy_name']

        if segment_id in sorted(dictionary_category.keys()):
            count = count + 1
            policy_name = re.sub('\d+_', '', policy_name)
            data.append({f"{policy_name}-{segment_id}": dictionary_category.get(count)})
    return data


# load all csv files to assign the data practice to its segment
def load_all_csv_files(path: str):
    data = []

    all_files = get_files_from_path(path)

    for file in all_files:
        file_basename = os.path.basename(file)
        data.append(assigning_category_to_seg(file_basename))

    return data


collect_category = load_all_csv_files('dataset/annotations')

# print(collect_category)

# collect the label data into a json
with open('collected_dp_category.json', 'w') as fp:
    json.dump(collect_category, fp)
