from csv import reader
from scipy import ndimage as ndi
from skimage import data, exposure
from skimage.filters import gabor_kernel
from skimage.util import img_as_float
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics.pairwise import kernel_metrics
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from misc_tools import images_in, flatten, get_train_test
from a_image_preprocessing import only_keep_every_third_pixel
from b_i_visual_feature_extraction import get_hog_train_test, get_gabor_train_test

# creates dataframe of features given function and image path
def create_feature_df(extraction_fuction, image_folder_name):
    df_from_csv = pd.read_csv(os.path.join(sys.path[0], 'train.csv')).fillna(0)
    
    feature_data = []
    for each_filename, each_image in images_in(image_folder_name, include_filename=True):
        preprocessed_image = only_keep_every_third_pixel(each_image)
        # makes dataframe row with filename and image features, and flattens each feature
        feature_data.append([each_filename] + flatten(extraction_fuction(preprocessed_image)))

    df_feat = pd.DataFrame(feature_data)
    df_feat_classified = df_feat.merge(
        df_from_csv[['filename', 'covid(label)']],
        left_on=0,
        right_on="filename"
    )
    df_feat_classified.drop([0, 'filename'], axis=1, inplace=True)

    return df_feat_classified

def visualize_features(name, features_df, labels_df, *other_args):
    '''
    ex:
        visualize_features("Gabor", *get_gabor_train_test())
        visualize_features("Hog", *get_hog_train_test())
    '''
    # add the label
    features_df = pd.DataFrame.copy(features_df)
    features_df['covid(label)'] = labels_df['covid(label)']
    
    # calculate evaluation metrics
    print(features_df.head())
    
    try:
        mutual_info_metric(features_df, f"{name} Mutual Info Metrics", )
    except Exception as error:
        print(f"Error when computing mutual_info_metric() for {name}")
        print(error)
    
    try:
        anova_metric(features_df, f"{name} Anova Metrics")
    except Exception as error:
        print(f"Error when computing anova_metric() for {name}")
        print(error)
        
    print("")


def chi2_metric(df_classified, title):
    """
    Calculates chi2 score of each feature with respect to the covid labels
    Plots results in a scatter plot
    """
    to_drop = [0, 'filename', 'covid(label)']
    to_drop = [item for item in to_drop if item in list(df_classified)]
    
    calc_chi2 = SelectKBest(score_func=chi2, k=4)
    df_chi2 = calc_chi2.fit(
        df_classified.drop(to_drop, axis=1),
        df_classified[['covid(label)']].values
    )
    plt.scatter(
        df_classified.drop(to_drop, axis=1).columns,
        df_chi2.scores_,
        alpha=0.3
    )
    plt.xlabel("Feature")
    plt.xticks(rotation=90)
    plt.ylabel("Chi2")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def mutual_info_metric(df_classified, title, to_drop=None, gen_image=True):
    """
    Calculates mutual_info score of each feature with respect to the covid labels
    Plots results in a scatter plot and image matrix of entropy
    """
    # remove all the string columns or later calculations will fail
    if to_drop is None:
        to_drop = [ each for each in list(df_classified.columns) if type(each) == str ]
    calc_mutual_info = SelectKBest(score_func=mutual_info_classif, k=1)
    df_classified = df_classified.dropna()
    # normalization of features in each sample
    feature_data = df_classified.drop(to_drop, axis=1).values
    mean_feature_data = np.mean(feature_data, axis=1)
    mean_feature_data = np.expand_dims(mean_feature_data, axis=1)
    feature_data = feature_data - mean_feature_data
    label = df_classified[['covid(label)']].values.ravel()

    df_mutual_info = calc_mutual_info.fit(feature_data, label)
    
    plt.figure()
    plt.scatter(
        df_classified.drop(to_drop, axis=1).columns,
        df_mutual_info.scores_,
        alpha=0.3
    )
    plt.xlabel("Feature")
    plt.xticks(rotation=90)
    plt.ylabel("mutual_info")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    if gen_image:
        try:
            # show the conditional entropy as an image matrix
            length = int(np.sqrt(df_mutual_info.scores_.shape[0]))
            tem = np.reshape(df_mutual_info.scores_, (length, length))
            plt.figure()
            plt.imshow(tem, cmap=plt.cm.gray)
            plt.title(title + "as image")
            plt.tight_layout()
            plt.show()
        except Exception as error:
            print(f"Error when trying to generate image for {title}")
            print(error)
            print()

def anova_metric(df_classified, title, to_drop=[0, 'filename', 'covid(label)'], gen_image=True):
    """
    Calculates anova score of each feature with respect to the covid labels
    Plots results in a scatter plot and image matrix of entropy
    """
    # drop all the string 
    to_drop = to_drop + [ each for each in list(df_classified.columns) if type(each) == str ]
    to_drop = [item for item in to_drop if item in list(df_classified)]
    calc_anova = SelectKBest(k=1)
    df_classified = df_classified.dropna()
    # normalization of features in each sample
    feature_data = df_classified.drop(to_drop, axis=1).values
    mean_feature_data = np.mean(feature_data, axis=1)
    mean_feature_data = np.expand_dims(mean_feature_data, axis=1)
    feature_data = feature_data - mean_feature_data
    label = df_classified[['covid(label)']].values.ravel()

    df_anova = calc_anova.fit(feature_data, label)
    plt.figure()
    plt.scatter(
        df_classified.drop(to_drop, axis=1).columns,
        df_anova.scores_,
        alpha=0.3
    )
    # plt.hlines(5, 0, len(df_classified.drop(
    #     to_drop, axis=1).columns))
    plt.xlabel("Feature")
    plt.xticks(rotation=90)
    plt.ylabel("anova")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    if gen_image:
        # show the conditional entropy as an image matrix
        length = int(np.sqrt(df_anova.scores_.shape[0]))
        tem = np.reshape(df_anova.scores_, (length, length))
        plt.figure()
        plt.imshow(tem, cmap=plt.cm.gray)
        plt.title(title + "as image")
        plt.tight_layout()
        plt.show()


def findConditionalEntropy(data, col):
    total = len(data)
    fractions = []
    ce = []
    for i in data[col].unique():
        bcount = data[data[col] == i]
        bcount = len(bcount[bcount[['covid(label)']] == 0])
        mcount = data[data[col] == i]
        mcount = len(mcount[mcount[['covid(label)']] == 1])
        part = bcount + mcount
        bpart = 0
        mpart = 0
        if part != 0:
            bpart = bcount / part
            mpart = mcount / part
        logb = 0
        logm = 0
        if bpart != 0:
            logb = np.log2(bpart)
        if mpart != 0:
            logm = np.log2(mpart)
        fractions.append(part)
        ce.append(-(bpart * logb + (mpart) * logm))
    fractions = np.array(fractions) / total
    return np.sum(fractions * np.array(ce))


def cond_entropy_metric(df_classified, title, to_drop=[0, 'filename', 'covid(label)'], genImage=True):
    cond_entropy_vals = []
    for i in df_classified.drop(
            to_drop, axis=1).columns:
        cond_entropy_vals.append(findConditionalEntropy(df_classified, i))

    plt.figure()
    plt.scatter(df_classified.drop(
        to_drop, axis=1).columns, cond_entropy_vals, alpha=0.3)
    plt.xlabel("Feature")
    plt.xticks(rotation=90)
    plt.ylabel("entropy")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def categorical_plots(data):
    '''
    Plots the categorical data from csv
    '''
    data = data.dropna()
    label_count = list(data.groupby(["covid(label)"]).count()["age"])
    plt.bar(["Negative", "Positive"], label_count)
    plt.tight_layout()
    plt.show()
    a = (data['age'] // 10 * 10).astype(int).astype(str) + \
        "-" + (data['age'] // 10 * 10 + 9).astype(int).astype(str)
    data.assign(age=a.to_numpy())
    for i in data.columns[1:-1]:
        pd.crosstab(data[i], data['covid(label)']).plot(
            kind='bar', stacked=False)
        plt.title(i)
        plt.legend(["Negative", "Positive"])
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def interpolate_categories(series):
    # from https://stackoverflow.com/questions/43586058/pandas-interpolate-with-nearest-for-non-numeric-values
    fact = series.astype('category').factorize()

    # get string as categorical (-1 is NaN)
    series_cat = pd.Series(fact[0]).replace(-1, np.nan)
    series_cat_interp = series_cat.interpolate(
        "nearest")  # interpolate categorical

    # dict connecting category to string
    cat_to_string = {i: x for i, x in enumerate(fact[1])}
    series_str_interp = series_cat_interp.map(
        cat_to_string)  # turn category back to string

    return series_str_interp
    
def interpolate_data(df):
    df[['age']] = df[['age']].fillna(df['age'].mean(skipna=True))
    df = get_countries(df)
    return df


def get_countries(df):
    # take the last word from the location values, to get the countries
    df = df.interpolate().apply(interpolate_categories)
    df['location'] = df['location'].str.split(",").str[-1].str.strip()
    df['covid(label)'] = df['covid(label)'].astype('category')
    return df


if __name__ == "__main__":
    train_features_df, train_labels_df, test_features_df = get_train_test()
    train_features_df = train_features_df.drop("images","columns")
    train_features_df['covid(label)'] = train_labels_df['covid(label)']
    train_features_df = interpolate_data(train_features_df)
    one_hot_csv = pd.get_dummies(train_features_df, columns=['gender', 'location'])
    
    print(train_features_df.head())
    # One hot encoding for countries and gender
    print("\nPlotted csv information with mutual_information")
    mutual_info_metric(one_hot_csv, "CSV Information", ['covid(label)'], False)
    print("Plotted csv information with chi2")
    chi2_metric(one_hot_csv, "CSV Information")
    print("\nPlotted csv information with anova")
    anova_metric(one_hot_csv, "CSV Information", ['covid(label)'], False)
    print("\nPlotting categorical data frequency bar charts...")
    categorical_plots(train_features_df)
    print("\nVisualizing HoG features...")
    visualize_features("HoG", *get_hog_train_test())
    print("\nVisualizing Gabor features...")
    visualize_features("Gabor", *get_gabor_train_test())
