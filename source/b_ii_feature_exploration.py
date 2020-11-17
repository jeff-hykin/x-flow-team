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

from skimage import data, exposure
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from sklearn.feature_selection import mutual_info_classif

from b_i_visual_feature_extraction import power, gabor_plot, hog_feature


def gabor_feature(image):
    '''
    image: the 2D image array

    Extract 40 Gabor features, 8 different direction and 5 different frequency
    '''
    results = []
    kernel_params = []
    # 8 direction
    for theta in range(8):
        theta = theta / 4. * np.pi
        # 5 frequency
        for frequency in range(1, 10, 2):
            frequency = frequency * 0.1
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (
                theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save power image for each image
            results.append(power(image, kernel))
    return results



# creates dataframe of features given function and image path
def create_feature_df(feature_function, name):
    df_from_csv = pd.read_csv(os.path.join(sys.path[0], 'train.csv')).fillna(0)
    image_path = name + '/'
    shrink = (slice(0, None, 3), slice(0, None, 3))
    feature_data = []
    for i in os.listdir(image_path):
        image_file = image_path + i
        image_file = cv2.imread(image_file, 0)
        image = img_as_float(image_file)[shrink]
        # makes dataframe row with filename and image features, and flattens each feature
        feature_data.append([i] + list(np.array(feature_function(image)).flatten()))

    df_feat = pd.DataFrame(feature_data)
    df_feat_classified = df_feat.merge(
        df_from_csv[['filename', 'covid(label)']], left_on=0, right_on="filename")
    return df_feat_classified


def visualizeFeatures(name, df_from_csv):
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
        gabor_data.append([i] + list(np.array(gabor_feature(image)).flatten()))
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
#     df_cond_entropy = calc_cond_entropy.fit(df_classified.drop(
#         to_drop, axis=1), df_classified[['covid(label)']].values.ravel())
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
    df_from_csv = pd.read_csv(os.path.join(sys.path[0], 'train.csv')).fillna(0)
    print(df_from_csv.head())
    # One hot encoding for countries and gender
    one_hot_csv = pd.get_dummies(df_from_csv, columns=['gender', 'location'])
    print("Plotted csv information with conditional entropy")
    conditional_entropy_metric(one_hot_csv, "CSV Information", ['filename', 'covid(label)'])
    visualizeFeatures("new_train100", df_from_csv)
