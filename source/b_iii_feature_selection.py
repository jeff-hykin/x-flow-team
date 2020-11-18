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

from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
import time

from misc_tools import split_data
from b_ii_feature_exploration import hog_feature, create_feature_df


# https://www.codespeedy.com/sequential-forward-selection-with-python-and-scikit-learn/
def sequential_forward_selection(data, num_tests, feature_counts):
	'''
	inputs: DataFrame including features and labels ('covid(label)'), 
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
							verbose=0,
							scoring='accuracy', # scored based on accuracy
							cv=num_tests) # num_tests-fold cross validaton
		# train
		sfs = sfs.fit(data_features, data_labels)
		print(type(sfs.k_feature_names_))

		# process results
		print('CV score for', nfeat, 'features:\n', sfs.k_score_)
		if sfs.k_score_ > best_features['accuracy']: # update best
			best_features = {'accuracy': sfs.k_score_, 'indicies': sfs.k_feature_names_, 'count': nfeat}
	
	print('Best SFS features accuracy is', best_features['accuracy'], 'with', best_features['count'], 'features at indicies', best_features['indicies'])
	return best_features['indicies']


# calculate fischer criterion value for feature d
def fischer_criterion(data, d):
	'''
	inputs: DataFrame including features and labels ('covid(label)'),
					feature 'd',
	output: fischer criterion score for feature 'd',
	'''
	# numer:
	# for each class
	# for each feature value in class
	# val - avg(all values of given feature in class)
	# denom:
	# for each class
	# (average of all feature values ever) - (average of all feature values in class)
	classes = data['covid(label)'].unique() # number of classes
	numerator = 0
	denomenator = 0
	ud = data[d].mean() # average of all values in feature / mu d
	for k in classes: # for each class / sum k
		vals_in_class = data.loc[data['covid(label)'] == k] # create subdf with all values in given class
		ukd = vals_in_class[d].mean() # average of feature values in given class / mu kd
		for xnd in vals_in_class[d]: # for each feature value in class / sum xn
			
			numerator += (xnd - ukd) ** 2
		denomenator += (ud - ukd) ** 2

	# avoid empty
	if denomenator == 0:
		fc = float('inf')
	else:
		# calculate final fischer's criterion
		fc = numerator / denomenator

	return fc

	

def fischer_criterion_selection(data, feature_counts):
	'''
	inputs: DataFrame including features and labels ('classes'),
					list with different numbers of features to test with,
	output: list of features chosen,
	'''
	# split training data into validation and training subsets
	train_ratio = 0.75 # ratio of training size to validation size
	data_split = split_data(data, train_ratio)
	# print(data_split[0][80])

	features_and_scores = [] # tuple of feature and score
	# calculate criterion for each feature
	features = data.keys()

	for feature in features:
		if feature != 'filename' and feature != 'covid(label)': # ignore filename and label
			features_and_scores.append(( feature , fischer_criterion(data_split[0],feature) ))

	# reorder array by increasing score (inplace)
	features_and_scores.sort(key=lambda tup: tup[1])

	print('Ordered', len(features_and_scores), 'features with fischer scores')
	# print(features_and_scores[0])
	# print(features_and_scores[1])

	# run validation subset for different feature counts
	val_labels = data_split[1]['covid(label)'].copy()
	train_labels = data_split[0]['covid(label)'].copy()
	best_features = {'accuracy': 0, 'count': 0}
	prev_nfeat = 1 # avoid starting with empty dataframe
	f = features_and_scores[0][0] # extract feature as f
	val_features = pd.DataFrame(data_split[1][f]) # make dataframe
	train_features = pd.DataFrame(data_split[0][f]) # make dataframe

	print('Testing for different numbers of features...')

	# test for different numbers of features
	for nfeat in feature_counts:
		print(nfeat, '...')

		# create features list for testing
		for ifeat in range(prev_nfeat, nfeat): # do not bother re-add already existing features
			f = features_and_scores[ifeat][0] # extract feature as f
			val_features = val_features.join(data_split[1][f]) # add to dataframe
			train_features = train_features.join(data_split[0][f]) # add to dataframe

		# test with validation set with nfeat features
		clf = LogisticRegression(random_state=0).fit(train_features, train_labels)
		score = clf.score(val_features, val_labels)
		print('CV score for', nfeat, 'features:\n', score)

		# if score is better, update
		if score > best_features['accuracy']:
			best_features = {'accuracy': score, 'count': nfeat}

		# update prev number of features for next iteration
		prev_nfeat = nfeat
		print('complete with', nfeat)

	print('Processing optimal features...', end=' ')
	# create array of optimal features
	optimal_features = []
	optimal_nfeats = best_features['count'] # best number of people
	for optimal_feature, optimal_score in features_and_scores[:optimal_nfeats]:
		optimal_features.append(optimal_feature)
	print('complete')

	print('Best Fischer Criterion features accuracy is', best_features['accuracy'], 'with', best_features['count'], 'features at indicies', optimal_features)
	return optimal_features

# main function
if __name__ == "__main__":

	print('Obtaining features...')

	# loading features df
	df_hog = create_feature_df('new_train100') # 1157 features...

	# # hyperparameter for both - list of feature counts to test
	feature_counts = [15, 25, 35, 40]


	print('Running feature selection algorithms...')
	# wrapper feature selection - sequential forward selection for multiple subsets
	num_tests = 5 # hyperparameter - cv folds
	sfs_features = sequential_forward_selection(df_hog, num_tests, feature_counts)

	# filter feature selection - fischer testing with different feature counts
	# fc_features = fischer_criterion_selection(df_hog, feature_counts)

	# print('Creating subfeature lists...')

	# create DataFrame for SFS feature choices
	df_sfs_X = pd.DataFrame(df_hog[sfs_features[0]])
	for feature in sfs_features[1:]:
		df_sfs_X = df_sfs_X.join(df_hog[feature]) # add to dataframe

	# create DataFrame for FC feature choices
	df_fc_X = pd.DataFrame(df_hog[fc_features[0]])
	if feature in fc_features:
		df_fc_X = df_fc_X.join(df_hog[feature]) # add to dataframe

	# create DataFrame for labels
	df_Y = df_hog['covid(label)'].copy()

	# run svm with cross validation for both sets of features
	print('Testing SVMs for SFS and FC...')
	clf_sfs = svm.SVC(kernel='linear', C=1)
	clf_fc = svm.SVC(kernel='linear', C=1)
	start_sfc = time.time()
	scores_sfs = cross_val_score(clf_sfs, df_sfs_X, df_Y, cv=5)
	end_sfc = time.time()
	scores_fc = cross_val_score(clf_fc, df_fc_X, df_Y, cv=5)
	end_fc = time.time()

	print('Completed cross validation with SFS features in', end_sfc - start_sfc, 'with average accuarcy of', np.average(scores_sfs))
	# print('Completed cross validation with FC features in', end_fc - end_sfc, 'with average accuarcy of', np.average(scores_fc))