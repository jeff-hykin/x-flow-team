from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import regex as re

from a_image_preprocessing import get_preprocessed_train_test
from b_i_visual_feature_extraction import hog_feature
from b_ii_feature_exploration import create_feature_df

def ada_boost(features, labels, name):
    '''
    features: a 2D (250, 10000 or xxx) numpy array of features
    labels: a 1D (250, 1) numpy array of label
    '''
    base_models = [
        (DecisionTreeClassifier(max_depth=1), 'SAMME.R'),
        (svm.SVC(kernel='linear', C=1), 'SAMME'),
        # (Perceptron(tol=1e-2, random_state=0), 'SAMME')
    ]
    fit_time = []
    avg_acc = []
    # try each base model
    for base_model in base_models:
        clf = AdaBoostClassifier(base_estimator=base_model[0], algorithm=base_model[1],
                                 n_estimators=100, random_state=0)
        # 5-fold cross validation
        cv_results = cross_validate(clf, features, labels, cv=5)
        fit_time.append(np.mean(cv_results['fit_time']))
        avg_acc.append(np.mean(cv_results['test_score']))
    
    ind = np.arange(len(base_models))  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, fit_time, width, color='SkyBlue', label='fit_time')
    
    # 
    # training time
    # 
    ax.set_ylabel('training time')
    ax.set_title(f'{name}: Training Time')
    plt.xticks(ind,('DecisionTree', 'SVM'))
    ax.legend()
    image_path = f"../graphs/{name}_ada_training_time"
    fig.savefig(image_path)
    print(f"saved image: {image_path}")

    # 
    # accuracy
    # 
    fig, ax = plt.subplots()
    rects2 = ax.bar(ind + width/2, avg_acc, width, color='IndianRed', label='avg_acc')
    ax.set_ylabel('Average Accuracy')
    ax.set_title(f'{name}: Average Accuracy')
    plt.xticks(ind,('DecisionTree', 'SVM'))
    ax.legend()
    image_path = f"../graphs/{name}_ada_training_time"
    fig.savefig(image_path)
    print(f"saved image: {image_path}")

    return fit_time, avg_acc

if __name__=="__main__":
    # just get the labels (same for all feature sets)
    _, labels_df, _ = get_preprocessed_train_test()
    
    # 
    # load in different feature sets
    # 
    list_of_data = []
    feature_folder = "./feature_selections/"
    for each_csv_name in os.listdir(feature_folder):
        features = pd.read_csv(feature_folder+each_csv_name)
        # remove the csv extension
        name = re.sub(r'\.csv$', "", each_csv_name)
        list_of_data.append((features, name))
    
    # 
    # convert data, run Ada Boost
    # 
    labels = labels_df.astype('float32')
    for features_df, name in list_of_data:
        print(f"running AdaBoost for {name}")
        # Run Adaboost here
        features = features_df.astype('float32')
        fit_time, avg_acc = ada_boost(features, labels, name)
