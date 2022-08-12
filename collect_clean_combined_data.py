import json

dataset_file_path = 'combined_data.json'
# data without 'bad' labels
clean_dataset_file_path = 'clean_combined_data.json'

with open(dataset_file_path) as fp:
    dataset_json = json.load(fp)


invalid_labels = {'The text introduces the policy, a section, or a group of practices, but it does not mention a specific practice.',
               'The text does not fit into our label scheme.',
               'The text describes a specific data practice that is not covered by our label scheme.',
               'A specific security measure not covered above.',
               'The policy makes a statement about how data from Californian users is treated (e.g., California privacy rights).',
               'The text describes how to contact the company with questions, concerns, or complaints about the privacy policy.',                     "When a change of an unspecified nature is made to the privacy policy, the policy date is updated or information about the change is posted as part of the policy. Users' choices regarding policy changes are not mentioned or are unclear.",
                "When a change of an unspecified nature is made to the privacy policy, the policy date is updated or information about the change is posted as part of the policy. Users can participate in a process to influence the change.",
                "When a change of an unspecified nature is made to the privacy policy, the policy date is updated or information about the change is posted as part of the policy. Users have no options regarding the change.",
                'The policy makes generic security statements, e.g., \"we protect your data\" or \"we use technology/encryption to protect your data\".'
                "When a change of an unspecified nature is made to the privacy policy, users are notified when visiting the website. Users' choices regarding policy changes are not mentioned or are unclear.",
                "When a change of an unspecified nature is made to the privacy policy, users are notified in an unspecified manner. Users' choices regarding policy changes are not mentioned or are unclear.",
                'The policy makes generic security statements, e.g., "we protect your data" or "we use technology/encryption to protect your data".',
                "The site collects your unspecified information for an unspecified purpose. Collection happens on the website.",
                "When a change of an unspecified nature is made to the privacy policy, users are notified when visiting the website. Users' choices regarding policy changes are not mentioned or are unclear."}

print(f"Original dataset has {len(dataset_json)} entries")
print(f"Size of invalid labels to remove is: {len(invalid_labels)}")
clean_dataset = []
datapoints_with_bad_labels = 0
occurances_of_bad_labels = 0
for datapoint in dataset_json:
    original_labels = datapoint['labels'].copy()
    datapoint['labels'] = [label for label in datapoint['labels'] if not label in invalid_labels]
    how_many_labels_we_removed = len(original_labels) - len(datapoint["labels"])
    if how_many_labels_we_removed > 0:
        datapoints_with_bad_labels += 1
        occurances_of_bad_labels += how_many_labels_we_removed
    if datapoint['labels']:
        clean_dataset.append(datapoint)

print(f"Removed occurances of bad labels: {occurances_of_bad_labels} times from {datapoints_with_bad_labels} datapoints")
print(f"Clean dataset has {len(clean_dataset)} entries")

with open(clean_dataset_file_path, "w") as f:
    json.dump(clean_dataset, f)
    print(f"Stored the clean dataset in {clean_dataset_file_path}")
