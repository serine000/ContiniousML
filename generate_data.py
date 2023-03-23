from utils import generate_random_abnormal_profile ,generate_random_normal_profile
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

# Generate data
train_dataframe = pd.DataFrame(columns=['connection_count', 'most_frequent_IP_connection_count', 'average_flow_duration', 'in_bytes', 'out_bytes',
             'direct_ip_access', 'flow_start_time', 'label'])
for i in range (0, 5000):
  train_dataframe.loc[len(train_dataframe)] = generate_random_abnormal_profile()

for i in range (0, 55000):
  train_dataframe.loc[len(train_dataframe)] = generate_random_normal_profile()

train_dataframe.loc[train_dataframe['label'] == 1, "attack"] = 1 
train_dataframe.loc[train_dataframe['label'] != 1, "attack"] = -1

target = train_dataframe['attack']
outliers = target[target == -1]
train_dataframe.drop(["label", "attack"], axis=1, inplace=True)


# Make a train/test split
train_data, test_data, train_target, test_target = train_test_split(train_dataframe, target, train_size = 0.8)

# Save it
if not os.path.isdir("data"):
    os.mkdir("data")

np.savetxt("data/train_features.csv", train_data)
np.savetxt("data/test_features.csv", test_data)
np.savetxt("data/train_labels.csv", train_target)
np.savetxt("data/test_labels.csv", test_target)
np.savetxt("data/target.csv", target)
np.savetxt("data/outliers.csv", outliers)