from csv import reader
from scipy import ndimage as ndi
from skimage import data, exposure
from skimage.filters import gabor_kernel
from skimage.util import img_as_float
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from misc_tools import split_data, get_train_test
from a_image_preprocessing import crop_resize
from b_i_visual_feature_extraction import power, gabor_plot, gabor_feature, hog_feature

# creates dataframe of features given function and image path
def create_feature_df(extraction_fuction, image_size):
    """
    has a lot of columns
    """
    train_features, train_labels, test_features = get_train_test()
    images = train_features['images'].values.tolist()
    
    preprocess_images = [ crop_resize(each, image_size) for each in images            ]
    feature_images    = [ extraction_fuction(each)      for each in preprocess_images ]
    feature_vectors   = [ each.flatten()                for each in feature_images    ]

    return pd.DataFrame(np.array(feature_vectors).transpose())

def visualize_features(name, df_from_csv):
    '''
    name : 'train' or 'test'
    '''
    shrink = (slice(0, None, 3), slice(0, None, 3))
    image_path = name + '/'
    gabor_data = []
    hog_data = []
    # for each image, get features and append to array
    for i in os.listdir(image_path):
        image_file = image_path + i
        image_file = cv2.imread(image_file, 0)
        image = img_as_float(image_file)[shrink]
        # makes dataframe row with filename and image features, and flattens each feature
        gabor_data.append([i] + list(np.array(gabor_feature(image)[0]).flatten()))
        hog_data.append([i] + list(np.array(hog_feature(image)).flatten()))
    # make dataframes and then calculate evaluation metrics
    df_gabor = pd.DataFrame(gabor_data)
    df_gabor_classified = df_gabor.merge(
        df_from_csv[['filename', 'covid(label)']], left_on=0, right_on="filename")
    print("Dataframe with Gabor features and classification")
    print(df_gabor_classified.head())
    conditional_entropy_metric(df_gabor_classified, "Gabor Metrics")
    # HoG metrics
    df_hog = pd.DataFrame(hog_data)
    df_hog_classified = df_hog.merge(
        df_from_csv[['filename', 'covid(label)']], left_on=0, right_on="filename")
    print("Dataframe with HoG features and classification")
    print(df_hog_classified.head())
    conditional_entropy_metric(df_hog_classified, "HoG Metrics")

def chi2_metric(df_classified, title):
    """
    Calculates chi2 score of each feature with respect to the covid labels
    Plots results in a scatter plot
    """
    calc_chi2 = SelectKBest(score_func=chi2, k=4)
    df_chi2 = calc_chi2.fit(df_classified.drop(
        [0, 'filename', 'covid(label)'], axis=1), df_classified[['covid(label)']].values)
    plt.scatter(df_classified.drop(
        [0, 'filename', 'covid(label)'], axis=1).columns, df_chi2.scores_, alpha=0.3)
    plt.xlabel("Feature")
    plt.ylabel("Chi2")
    plt.title(title)
    plt.show()

def conditional_entropy_metric(df_classified, title, to_drop=[0, 'filename', 'covid(label)']):
    """
    Calculates cond_entropy score of each feature with respect to the covid labels
    Plots results in a scatter plot and image matrix of entropy
    """
    calc_cond_entropy = SelectKBest(score_func=mutual_info_classif, k=4)
    
    # normalization of features in each sample
    feature_data = df_classified.drop(to_drop, axis=1).values
    mean_feature_data = np.mean(feature_data, axis=1)
    mean_feature_data = np.expand_dims(mean_feature_data, axis=1)
    feature_data = feature_data - mean_feature_data
    label = df_classified[['covid(label)']].values.ravel()

    df_cond_entropy = calc_cond_entropy.fit(feature_data, label)
    print(df_cond_entropy.scores_)
    plt.figure()
    plt.scatter(df_classified.drop(
        to_drop, axis=1).columns, df_cond_entropy.scores_, alpha=0.3)
    plt.xlabel("Feature")
    plt.ylabel("cond_entropy")
    plt.title(title)
    plt.show()
    
    # show the conditional entropy as an image matrix
    length = int(np.sqrt(df_cond_entropy.scores_.shape[0]))
    tem = np.reshape(df_cond_entropy.scores_, (length, length))
    plt.figure()
    plt.imshow(tem, cmap=plt.cm.gray)
    plt.title(title + "as image")
    plt.show()


if __name__ == "__main__":
    train_features, train_labels, test_features = get_train_test()
    print(train_features.columns)
    # df_from_csv = pd.read_csv(os.path.join(sys.path[0], 'train.csv')).fillna(0)
    # print(df_from_csv.head())
    # # One hot encoding for countries and gender
    # one_hot_csv = pd.get_dummies(df_from_csv, columns=['gender', 'location'])
    # print("Plotted csv information with conditional entropy")
    # conditional_entropy_metric(one_hot_csv, "CSV Information", ['filename', 'covid(label)'])
    # visualize_features("new_train100", df_from_csv)
