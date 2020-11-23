from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from b_i_visual_feature_extraction import hog_feature
from b_ii_feature_exploration import create_feature_df
import numpy as np

def ada_boost(features, labels):
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
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('training time')
    ax.set_title('training time for each base model of AdaBoost')
    plt.xticks(ind,('DecisionTree', 'SVM'))
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    rects2 = ax.bar(ind + width/2, avg_acc, width, color='IndianRed', label='avg_acc')
    ax.set_ylabel('Average accuracy')
    ax.set_title('Average accuracy for each base model of AdaBoost')
    plt.xticks(ind,('DecisionTree', 'SVM'))
    ax.legend()
    plt.show()

    return fit_time, avg_acc

if __name__=="__main__":
    # wrapper features

    # filter features

    # loading features df
    df_hog = create_feature_df(hog_feature, 'new_train100') # 10000 features...

    df_hog = df_hog.values
    hog_feature = df_hog[:, 1:-2].astype('float32')
    hog_label = df_hog[:, -1].astype('float32')
    fit_time, avg_acc = ada_boost(hog_feature, hog_label)

