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

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier as KNN

from misc_tools import split_data
from b_i_visual_feature_extraction import hog_feature, create_feature_df


def sequential_forward_selection(data, num_tests, feature_counts):
	'''
	inputs: DataFrame including features and labels ('classes'), 
					(hyperparam) number of iterations of SFS to be ran,
					(hyperparam) number of features,
	output: list of features chosen,
	'''
	print('Running SFS...')
	# split data into features and labels
	data_labels = data['covid(label)'].copy()
	data_features = data.copy()
	data_features = data_features.drop(columns='covid(label)')


	# keep track of best features' accuracies, indicies (of the chosen features) in the DataFrame, and number of features
	best_features = {'accuracy': 0, 'indicies': [], 'count': 0}

	# test with different numbers of features
	for nfeat in feature_counts:
		# sequential forward selection
		knn = KNN(n_neighbors=3) # n neighbors could be a hyperparameter, but let's keep it to 3
		sfs = SFS(knn, # estimator to use. RFS also seen
							k_features=nfeat, # number of features to be chosen
							forward=True, # SFS, not SBS
							floating=False,
							verbose=2,
							scoring='accuracy', # scored based on accuracy
							cv=num_tests) # num_tests-fold cross validaton
		# train
		sfs = sfs.fit(data_features, data_labels)

		# process results
		print('CV score for', nfeat, 'features:\n', sfs.k_score_)
		if sfs.k_score_ > best_features['accuracy']: # update best
			best_features = {'accuracy': sfs.k_score_, 'indicies': sfs.k_feature_idx_, 'count': nfeat}
	
	print('Best SFS features accuracy is', best_features['accuracy'], 'with', best_features['count'], 'features at indicies', best_features['indicies'])
	return best_features['indicies']


# calculate fischer criterion value for feature d
def fischer_criterion(data, d):
	'''
	inputs: DataFrame including features and labels ('classes'),
					feature 'd',
	output: fischer criterion score for feature 'd',
	'''
	classes = train_labels.ix[:,0].unique() # number of classes
	for k, _class in enumerate(classes): # for each class
		average = 1
	return 1

	

def fischer_criterion_search(data, feature_counts):
	'''
	inputs: DataFrame including features and labels ('classes'),
					list with different numbers of features to test with,
	output: list of features chosen,
	'''
	# split training data into validation and training subsets
	train_ratio = 0.8 # ratio of training size to validation size
	data_split = split_data(data, 2, training_ratio)

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

	# loading features df
	df_hog = create_feature_df(hog_feature, 'new_train100') # 1157 features...

	# hyperparameter for both - list of feature counts to test
	feature_counts = [15, 25, 35, 40]

	# wrapper feature selection - sequential forward selection for multiple subsets
	num_tests = 10 # hyperparameter - cv folds
	sequential_forward_selection(df_hog, num_tests, feature_counts)

	# filter feature selection - fischer testing with different feature counts
	fischer_criterion_search(df_hog, feature_counts)