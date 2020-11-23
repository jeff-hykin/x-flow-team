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

import multiprocessing

from misc_tools import split_data, auto_cache
from b_i_visual_feature_extraction import hog_feature, get_hog_train_test, get_gabor_train_test
from b_ii_feature_exploration import create_feature_df


# https://www.codespeedy.com/sequential-forward-selection-with-python-and-scikit-learn/
def sequential_forward_selection(data, num_tests, feature_counts):
    '''
    inputs: DataFrame including features and labels ('covid(label)'), 
                    (hyperparam) number of iterations of SFS to be ran,
                    (hyperparam) number of features,
    output: list of features chosen,
    '''
    global compute_sfs # a bad hack for getting mutliprocessing working
    print('Running SFS...')
    # split data into features and labels
    data_labels = data['covid(label)'].copy()
    data_features = data.copy().drop(columns='covid(label)')

    # create a pure function (all values as arguments)
    def compute_sfs(arg):
        number_of_features, num_tests = arg
        # sequential forward selection
        sfs = SFS(
            KNN(n_neighbors=3), # estimator to use. RFS also seen
            k_features=number_of_features, # number of features to be chosen
            forward=True, # SFS, not SBS
            floating=False,
            verbose=0,
            scoring='accuracy', # scored based on accuracy
            cv=num_tests # num_tests-fold cross validaton
        ).fit(data_features, data_labels) # train
        return sfs.k_score_, sfs.k_feature_names_, number_of_features
    
    # use multiprocessing to get the work done fast(er)
    number_of_processes = (multiprocessing.cpu_count()-1) or 1
    with multiprocessing.Pool(number_of_processes) as process_pool:
        argument_list = [ (number_of_features, num_tests) for number_of_features in feature_counts ]
        results = process_pool.map(compute_sfs, argument_list)
    
    # keep track of best features' accuracies, indicies (of the chosen features) in the DataFrame, and number of features
    best_features = {'accuracy': 0, 'indicies': [], 'count': 0}
    # see which one was the best
    for score, feature_names, number_of_features in results:
        print('CV score for', number_of_features, 'features: ', score)
        if score > best_features['accuracy']: # update best
            best_features = {'accuracy': score, 'indicies': feature_names, 'count': number_of_features}
    
    print('Best SFS accuracy is', best_features['accuracy'], 'with', best_features['count'], 'features')
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
    train_data, validation_data = split_data(data, train_ratio)

    features_and_scores = [] # tuple of feature and score
    # calculate criterion for each feature
    features = data.keys()
    for feature in features:
        if feature != 'filename' and feature != 'covid(label)': # ignore filename and label
            features_and_scores.append(( feature , fischer_criterion(train_data, feature) ))

    # reorder array by increasing score (inplace)
    features_and_scores.sort(key=lambda tup: tup[1])
    print('Ordered', len(features_and_scores), 'features with fischer scores')

    # run validation subset for different feature counts
    val_labels   = validation_data['covid(label)'].copy()
    train_labels = train_data['covid(label)'].copy()
    best_features = {'accuracy': 0, 'count': 0}
    previous_number_of_features = 1 # avoid starting with empty dataframe
    feature, score = features_and_scores[0] # extract feature as f
    val_features   = pd.DataFrame(validation_data[feature]) # make dataframe
    train_features = pd.DataFrame(train_data[feature]) # make dataframe

    # test for different numbers of features
    print('Testing for different numbers of features...')
    for number_of_features in feature_counts:
        # create features list for testing
        for feature_index in range(previous_number_of_features, number_of_features): # do not bother re-add already existing features
            feature, score = features_and_scores[feature_index] # extract feature as f
            val_features   = val_features.join(validation_data[feature]) # add to dataframe
            train_features = train_features.join(train_data[feature]) # add to dataframe

        # test with validation set with number_of_features features
        clf = LogisticRegression().fit(train_features, train_labels)
        score = clf.score(val_features, val_labels)
        print('CV score for', number_of_features, 'features:', score)

        # if score is better, update
        if score > best_features['accuracy']:
            best_features = {'accuracy': score, 'count': number_of_features}

        previous_number_of_features = number_of_features

    print('Processing optimal features...', end=' ')
    # create array of optimal features
    optimal_features = []
    optimal_number_of_featuress = best_features['count'] # best number of people
    for optimal_feature, optimal_score in features_and_scores[:optimal_number_of_featuress]:
        optimal_features.append(optimal_feature)
    print('complete')

    print('Best Fischer Criterion accuracy is', best_features['accuracy'], 'with', best_features['count'], 'features')
    return optimal_features

def filter_process(features_with_labels_df, feature_counts):
    print('#')
    print('#  Filter: Fischer Criterion')
    print('#')
    # filter feature selection - fischer testing with different feature counts
    filter_features = fischer_criterion_selection(features_with_labels_df, feature_counts)
    # create DataFrame for FC feature choices
    optimized_filter_df = pd.DataFrame({})
    for feature in filter_features:
        optimized_filter_df[feature] = features_with_labels_df[feature]
    
    return filter_features, optimized_filter_df

def wrapper_process(features_with_labels_df, feature_counts):
    # wrapper feature selection - sequential forward selection for multiple subsets
    print('#')
    print('#  Wrapper: SFS')
    print('#')
    num_tests = 5 # hyperparameter - cv folds
    wrapper_features = sequential_forward_selection(features_with_labels_df, num_tests, feature_counts)
    
    # this 'if' happens when the process is interupted
    if len(wrapper_features) == 0:
        exit()
        
    print('wrapper_features = ', wrapper_features)
    optimized_wrapper_df = pd.DataFrame({})
    for feature in wrapper_features:
        optimized_wrapper_df[feature] = features_with_labels_df[feature]
    
    return wrapper_features, optimized_wrapper_df

def compare(selected_features_1_df, selected_features_2_df, labels_df):
    print('#')
    print('#  Wrapper vs Filter')
    print('#')
    
    # run svm with cross validation for both sets of features
    clf_sfs    = svm.SVC(kernel='linear', C=1)
    clf_fc     = svm.SVC(kernel='linear', C=1)
    
    # sfs
    start  = time.time()
    scores_sfs = cross_val_score(clf_sfs, selected_features_1_df, labels_df, cv = 5)
    sfs_time = time.time() - start
    # fc 
    start    = time.time()
    scores_fc  = cross_val_score(clf_fc , selected_features_2_df, labels_df, cv = 5)
    fc_time = time.time() - start
    
    return sfs_time, scores_sfs, fc_time, scores_fc

def run_filter_vs_wrapper_competition(features_df, labels_df, feature_counts):
    # add label to the features since that's what some other functions expect
    features_df = features_df.copy()
    features_df['covid(label)'] = labels_df['covid(label)']
    
    # compute which features are the most helpful
    filter_features, optimized_filter_df = auto_cache(filter_process, features_df, feature_counts)
    print('filter_features = ', filter_features)
    wrapper_features, optimized_wrapper_df = auto_cache(wrapper_process, features_df, feature_counts)
    print('wrapper_features = ', wrapper_features)
    
    # see which performed better in cross validation using an SVM model
    sfs_time, scores_sfs, fc_time, scores_fc = compare(
        selected_features_1_df=optimized_filter_df,
        selected_features_2_df=optimized_wrapper_df,
        labels_df=train_labels_hog,
    )
    print('Fischer Criterion')
    print('    score: ', np.average(scores_fc))
    print('    time: ', fc_time)
    print('Sequential Forward Selection')
    print('    score: ', np.average(scores_sfs))
    print('    time: ', sfs_time)
    
    # results
    return (
        filter_features,
        optimized_filter_df,
        
        wrapper_features,
        optimized_wrapper_df,
        
        sfs_time,
        scores_sfs,
        fc_time,
        scores_fc
    )
    
# main function
if __name__ == "__main__":
    print('#')
    print('#  Loading Data (~5min max time)')
    print('#')
    # these handle image preprocessing and one-hot encoding
    train_features_hog, train_labels_hog, test_features_hog = auto_cache(get_hog_train_test)
    train_features_gabor, train_labels_gabor, test_features_gabor = auto_cache(get_gabor_train_test)
    
    # compute which features are the most helpful
    feature_counts = [ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 25, 27, 35, 40 ]
    hog_results    = auto_cache(run_filter_vs_wrapper_competition, train_features_hog,   train_labels_hog,   feature_counts)
    gabor_results  = auto_cache(run_filter_vs_wrapper_competition, train_features_gabor, train_labels_gabor, feature_counts)