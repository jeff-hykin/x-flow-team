import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
import cv2
from csv import reader
from skimage.util import img_as_float
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from misc_tools import get_train_test
from b_i_visual_feature_extraction import hog_feature, create_feature_df


def sequential_forward_selection(train_inputs, train_labels, features, num_tests):
	# split data into num_tests sets
	input_list, label_list = split_data(train_inputs, train_labels, num_tests)

	# run sequential forward selection num_tests times
	# for i in range(num_tests):
		# sequential forward selection
	

	return 1


# calculate fischer criterion value for feature d
def fischer_criterion(train_labels, d):
	classes = train_labels.ix[:,0].unique() # number of classes
	for k, _class in enumerate(classes): # for each class
		average = 

	

def fischer_criterion_search(train_inputs, train_labels, features, feature_counts):
	# split training data into validation and training subsets
	train_ratio = 0.8 # ratio of training size to validation size
	input_list, label_list = split_data(train_inputs, train_labels, 2, training_ratio)

	features_and_scores = [] # tuple of feature and score
	# calculate criterion for each feature
	for feature in range(features):
		features_and_scores.append((feat, fischer_criterion(label_list[0], feat)))

	# reorder array by increasing score (inplace)
	features_and_scores.reorder(key=lambda tup: tup[1])

	# run validation subset for different feature counts
	best_nfeat = 0
	# for nfeat in range(feature_counts):
		# test with validation set with x features
		
	return 1

# main function
if __name__ == "__main__":

	# loading data and features
	train_inputs, train_labels, test_inputs = get_train_test()
	
	features = [] # TODO: calculate features (46241 featuresss)

	# wrapper feature selection - sequential forward selection for multiple subsets
	num_tests = 10 # hyperparameter - number of times we run algorithm with subsets of data
	sequential_forward_selection(train_inputs, train_labels, features, num_tests)

	# filter feature selection - fischer testing with different feature counts
	feature_counts = [5, 10, 15, 20, 25] # hyperparameter - list of feature counts to test
	fischer_criterion_search(train_inputs, train_labels, features, feature_counts)