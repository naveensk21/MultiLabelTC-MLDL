import os.path
import json

# load the label dataset
label_dataset_file_path = 'collected_labels.json'

with open(label_dataset_file_path) as f:
  label_dataset = json.load(f)


# load the policy text from sanitized_policies
def load_policy_by_name(name) -> str:
  policy_dir = "dataset/sanitized_policies"

  policy_filenames = os.listdir(policy_dir)
  for file_name in policy_filenames:
    if name in file_name:
      policy_path = f"{policy_dir}/{file_name}"
      with open(policy_path) as f:
        return f.read()


# load the all the policy text per policy name and its segments
def load_policy_segment(policy_name, segment_idx):
  policy_text = load_policy_by_name(policy_name)
  policy_text_segment = policy_text.split("|||")[segment_idx]
  return policy_text_segment


# assign the policy text to said segment per file with key: policyname-segment_id
# value: policy text found at the said segment
def policy_text_list(policy_name):
  policy_text_list = []
  policy_text = load_policy_by_name(policy_name)
  policy_text_seg = policy_text.split('|||')

  count = 0 - 1
  for i in policy_text_seg:
    count = count + 1
    policy_text_list.append(({f'{policy_name}-{count}': policy_text_seg[count],
                            # 'segment_id': count,
                            # 'policy_text': policy_text_seg[count]
                              }))

  return policy_text_list


# load all the policy text for all the files
def load_all_policy_text(path: str):
  data = []
  all_files = os.listdir(path)
  for file in all_files:
    name = os.path.basename(file).split('.')[0]
    policy_name = name.split('_')[1]

    all = policy_text_list(policy_name)

    data.append(all)

  return data

collect_policy_text = load_all_policy_text('dataset/sanitized_policies')
#
# with open('collected_policy_text.json', 'w') as fp:
#   json.dump(collect_policy_text, fp)







#============================== Testting ground =======================================
# Try for a single file
# for testing - zacks policy text
zacks_policy_text = policy_text_list('zacks')

#zacks label dataset
with open('zacks_label.json') as fp:
  zacks_label = json.load(fp)

# search through the policy_text_dataset's key and match it to the
# key of the label dataset
# if a match is found, add the policy text to the location of that match

# key and values for the zacks policy_text data
policy_text_keys = [key for d in zacks_policy_text for key in d.keys()]
policy_text_values = [value for d in zacks_policy_text for value in d.values()]


# search for keys in zacks policy text
def search_add():
  output = []
  count = 0 - 1
  for element in zacks_label:
    count = count + 1
    label_keyname = f'{element["policy_name"]}-{str(element["segment_id"])}'
    if label_keyname in policy_text_keys:
      element['policy_text'] = policy_text_values[count]
      output.append(element)
  return output

# print(search_add())




















# gets the key from policy_text
def get_policyname_from_policy_text():
  with open('collected_policy_text.json') as fp:
    policy_text_data = json.load(fp)

  for entry in policy_text_data:
    for item in entry:
      for key in item.keys():
        return key






# load zacks.com.csv label dataset
zacks_label_path = 'zacks_label.json'

with open(zacks_label_path) as f:
  zacks_label_dataset = json.load(f)


# test only for zacks.com.csv file
def policy_text_to_labels(policy_name):
  policy_text = load_policy_by_name(policy_name)
  policy_text_seg = policy_text.split('|||')

  for index in range(len(zacks_label_dataset)):
    if policy_name in zacks_label_dataset[index]['policy_name']:
      zacks_label_dataset[index]['policy_text'] = policy_text_seg[index]

  return zacks_label_dataset


# test to assign the policy text to the correct policy name and segment
def all_policy_text_to_label(policy_name):
  policy_text = load_policy_by_name(policy_name)
  policy_text_seg = policy_text.split('|||')

  for index in range(len(zacks_label_dataset)):
    if policy_name in label_dataset[index]['policy_name']:
      zacks_label_dataset[index]['policy_text'] = policy_text_seg[index]

  return zacks_label_dataset

#============================== End Testing ground =======================================




