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
        training_inputs["images"] = training_inputs["filename"].transform(
            # convert to grayscale because they're basically already grayscale
            lambda each_filename: cv2.imread(relative_path("train", each_filename), 0)
        )
        training_inputs = training_inputs.drop("filename", axis="columns")
        
        # 
        # Test Data
        # 
        # filename, gender, age, location
        test_inputs = pd.read_csv(os.path.join(sys.path[0], 'test.csv')).fillna(0)
        # test_labels = None, yup no testing labels
        test_inputs["images"] = test_inputs["filename"].transform(
            # convert to grayscale because they're basically already grayscale
            lambda each_filename: cv2.imread(relative_path("test", each_filename), 0)
        )
        test_inputs = test_inputs.drop("filename", axis="columns")
        
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

# splits train data into multiple subsets
def split_data(data_inputs, data_labels, num_subsets, ratio=-1):
    # combine inputs and labels, reorder, then seperate again
    label_title = data_labels.keys()[0]
    data = pd.append(data_inputs, data_labels)
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)
    data_inputs = data.drop(columns=label_title)
    data_labels = data[label_title]

    num_points = len(data_labels[label_title])
    input_split = []
    label_split = []

    # if ratio is -1, then use even ratios
    if ratio == -1:
        cut_size = (math.ceil(num_points - first_cut))/num_subsets
        start_point = 0
        end_point = cut_size

        # process cuts
        for a in range(num_subsets-1):
            input_split.append(data_inputs[start_point:end_point].copy())
            label_split.append(data_labels[start_point:end_point].copy())
            start_point = end_point
            end_point += cut_size

        # process last cut
        start_point = end_point
        end_point = num_points
        input_split.append(data_inputs[start_point:end_point].copy())
        label_split.append(data_labels[start_point:end_point].copy())

    # make first cut ratio size, and rest evenly sized
    else:
        first_cut = math.ceil(ratio * num_points)
        other_cut = (math.ceil(num_points - first_cut))/(num_subsets-1)
        start_point = 0
        end_point = first_cut

        # process first cut
        input_split.append(data_inputs[start_point:end_point].copy())
        label_split.append(data_labels[start_point:end_point].copy())

        
        # process intermediate cuts
        for a in range(1, num_subsets-1):
            start_point = end_point
            end_point += other_cut
            input_split.append(data_inputs[start_point:end_point].copy())
            label_split.append(data_labels[start_point:end_point].copy())

        # process last cut
        start_point = end_point
        end_point = num_points
        input_split.append(data_inputs[start_point:end_point].copy())
        label_split.append(data_labels[start_point:end_point].copy())

        
    return (input_split, label_split)