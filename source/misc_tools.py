from collections import Counter # frequency count
from csv import reader
from scipy import ndimage as ndi
from skimage import data, exposure
from skimage.filters import gabor_kernel
from skimage.util import img_as_float
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

relative_path = lambda *filepath_peices : os.path.join(os.path.dirname(__file__), *filepath_peices)

data = ()
def get_train_test():
    """
    returns train_features, train_labels, test_features
    They're dataframes
    Also the train_features["images"] are the list of images
    """
    global data
    # if data hasn't been loaded
    if len(data) == 0:
        # 
        # Train data
        # 
        # filename, gender, age, location, covid(label)
        training_df = pd.read_csv(os.path.join(sys.path[0], 'train.csv')).fillna(0)
        training_labels = training_df["covid(label)"]
        training_inputs = training_df.drop("covid(label)", axis="columns")
        training_inputs["images"] = training_inputs["filenames"].transform(
            # convert to grayscale because they're basically already grayscale
            lambda each_filename: cv2.cvtColor(cv2.imread(relative_path("train", each_filename), 0), cv2.COLOR_BGR2GRAY)
        )
        training_inputs = training_inputs.drop("filenames", axis="columns")
        
        # 
        # Test Data
        # 
        # filename, gender, age, location
        test_inputs = pd.read_csv(os.path.join(sys.path[0], 'test.csv')).fillna(0)
        # test_labels = None, yup no testing labels
        test_inputs["images"] = test_inputs["filenames"].transform(
            # convert to grayscale because they're basically already grayscale
            lambda each_filename: cv2.cvtColor(cv2.imread(relative_path("test", each_filename), 0), cv2.COLOR_BGR2GRAY)
        )
        test_inputs = training_inputs.drop("filenames", axis="columns")
        
        data = (training_inputs, training_labels, test_inputs)
    
    return data

def conditional_entropy(feature_data, labels):
    if type(feature_data) == dict:
        feature_names = feature_data.keys()
    else:
        feature_names = range(len(feature_data))
    
    # this should end up being set([ True, False ])
    label_values = set(labels)
    conditional_entropy = {}
    for each_feature in feature_names:
        total_samples_for_this_feature = len(feature_data[each_feature])
        not_usefulness = 0
        # ocurrance-count for each value in this feature
        feature_value_count = dict(Counter(feature_data[each_feature]))
        # ocurrance-count for each value+outcome in this feature (more keys, each key is a tuple)
        feature_count = dict(Counter(zip(feature_data[each_feature], labels)))
        for each_feature_value, number_of_samples_with_feature_value in feature_value_count.items():
            def calculate_not_usefulness(label):
                # number of features that had this value and this label
                count_for_this_label = feature_count.get((each_feature_value, label), 0)
                label_proportion_for_feature_value = count_for_this_label/number_of_samples_with_feature_value
                if count_for_this_label > 0:
                    return label_proportion_for_feature_value * math.log2(label_proportion_for_feature_value)
                else:
                    return 0
            
            feature_value_proportion = number_of_samples_with_feature_value / total_samples_for_this_feature
            unscaled_not_usefulness = sum([ calculate_not_usefulness(each) for each in label_values ])
            not_usefulness -= feature_value_proportion * unscaled_not_usefulness

        conditional_entropy[each_feature] = not_usefulness
    
    return conditional_entropy



train_features, train_labels, test_features = get_train_test()

plt.figure()
plt.imshow(train_features["images"][0], cmap=plt.cm.gray)
plt.title("image")
plt.show()