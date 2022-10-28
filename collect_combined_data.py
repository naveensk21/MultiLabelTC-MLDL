import json

# load labelled data and policy text data
with open('collected_labels.json') as fp:
    labelled_data = json.load(fp)

with open('collected_policy_text.json') as fp:
    policy_txt_data = json.load(fp)


# gets all the keys from policy text dataset
def get_keys_policy_txt():
    return [keys for entry in policy_txt_data for item in entry for keys in item.keys()]


# gets the value from the key in policy text data
def get_value_from_key(key_name):
    for entry in policy_txt_data:
        for item in entry:
            if key_name in item.keys():
                get_value = item.get(key_name)
                return get_value


# add the policy texts to the labels with the corresponding segment number
def combined_data():
    data = []
    for list1 in labelled_data:
        for entry in list1:
            label_key = f'{entry["policy_name"]}-{str(entry["segment_id"])}'
            if label_key in get_keys_policy_txt():
                entry['policy_text'] = get_value_from_key(label_key)
                data.append(entry)
    return data


joined_data = combined_data()

# dump the data into json file
with open('combined_data.json', 'w') as fp:
    json.dump(joined_data, fp)

















