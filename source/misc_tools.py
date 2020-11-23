from collections import Counter # frequency count
import numpy
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
def get_train_test(train_path="train", test_path="test"):
    """
    ex:
        train_features, train_labels, test_features = get_train_test()
    
    returns train_features, train_labels, test_features
    They're dataframes
    Also train_features["images"] is a list of images
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
            lambda each_filename: cv2.imread(relative_path(train_path, each_filename), 0)
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
            lambda each_filename: cv2.imread(relative_path(test_path, each_filename), 0)
        )
        test_inputs = test_inputs.drop("filename", axis="columns")
        
        data = (training_inputs, pd.DataFrame(training_labels), test_inputs)
    
    # always return a copy
    return tuple([ pd.DataFrame.copy(each) for each in data])

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
# can be used to make train/val split, or for cross validation
def split_data(data, ratio=0.5):
    '''
    inputs: DataFrame data,
            number of subsets to be created,
            ratio of first subset length to total length,
    output: array of subsets
    '''
    # print(data.head())
    # combine inputs and labels, reorder, then seperate again
    data = data.sample(frac=1)
    data = data.reset_index(drop=True)

    num_points = len(data.index)
    data_split = []

    # print('num_points:', num_points)

    # make first cut ratio size, and rest evenly sized
    first_cut = math.ceil(ratio * num_points)
    # print('first cut', first_cut)
    start_point = 0
    end_point = first_cut

    # process first cut
    data_split.append(data[start_point:end_point].copy())

    # process last cut
    start_point = end_point
    end_point = num_points
    data_split.append(data[start_point:end_point].copy())

    
    # print(data_split[0][100])

        
    return data_split

def flatten(iterable):
    return list(np.array(iterable).flatten())

def images_in(foldername, include_filename=False):
    """
    returns [image1,image2, ...] (each image is a numpy array)
    if include_filename=True
        returns [ (filename1, image1), (filename2, image2), ... ]
    """
    images = []
    for each_image_filename in os.listdir(foldername):
        image_path = os.path.join(foldername, each_image_filename)
        if include_filename:
            images.append((each_image_filename, cv2.imread(image_path, 0)))
        else:
            images.append(cv2.imread(image_path, 0))
    return images

def is_grayscale(image):
    as_numpy_array = numpy.array(image)
    if len(as_numpy_array.shape) == 3:
        if as_numpy_array.shape[2] >= 3:
            return False
    return True

def split_into_columns(dataframe, column_name):
    feature_list = dataframe[column_name].tolist()
    new_column_names = list(range(len(feature_list[0])))
    dataframe[new_column_names] = pd.DataFrame(feature_list, index=dataframe.index)
    new_dataframe = pd.DataFrame(dataframe[column_name].to_list(), columns=new_column_names)
    for each_column in dataframe.drop(column_name, "columns").columns:
        new_dataframe[each_column] = dataframe[each_column]
    return new_dataframe

def list_of_images_to_dataframe(list_of_images):
    flattened = [ np.array(each_image).flatten() for each_image in list_of_images ]
    big_array = np.array(flattened)
    dict_of_feature_lists = {
        each_index: each_feature_colum for each_index, each_feature_colum in enumerate(big_array.transpose())
    }
    # here's what that'd look like
    # (each image is visually column here)
    #         image1, image2, image3, ...
    # {
    #    0: [ 0.324,  0.324,  0.324,  ... ]
    #    1: [ 0.324,  0.324,  0.324,  ... ]
    #    2: [ 0.324,  0.324,  0.324,  ... ]
    #    3: [ 0.324,  0.324,  0.324,  ... ]
    #    4: [ 0.324,  0.324,  0.324,  ... ]
    # }
    # but in the data frame the 0: 1: 2: 3: will be the columns, with each row being a feature
    return pd.DataFrame(dict_of_feature_lists)

def large_pickle_load(file_path):
    """
    This is for loading really big python objects from pickle files
    ~4Gb max value
    """
    import pickle
    import os
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

def large_pickle_save(variable, file_path):
    """
    This is for saving really big python objects into a file
    so that they can be loaded in later
    ~4Gb max value
    """
    import pickle
    bytes_out = pickle.dumps(variable, protocol=4)
    max_bytes = 2**31 - 1
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


# just a self-made fix for unhashable builtin types
def hash_decorator(hash_function):
    import collections

    def is_iterable(thing):
        # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
        try:
            iter(thing)
        except TypeError:
            return False
        else:
            return True
            
    def make_hashable(value):
        type_of_value = type(value)
        output = None
        if type_of_value == str or type_of_value == frozenset:
            output = value
        elif type_of_value == set:
            output = frozenset([ make_hashable(each) for each in value ])
        elif type_of_value == dict:
            sorted_iterable = list(value.items())
            sorted_iterable.sort()
            output = tuple([ make_hashable(each) for each in sorted_iterable ])
        elif type_of_value == pd.core.frame.DataFrame:
            value_as_string = value.to_csv()
            output = hash(value_as_string)
        elif is_iterable(value):
            output = tuple([ make_hashable(each) for each in value ])
        else:
            output = value
        return output
        
    def wrapper(*args, **kwargs):
        try:
            return hash_function(*args, **kwargs)
        except:
            if len(args) == 1 and len(kwargs) == 0:
                hashable_argument = make_hashable(args[0])
                hashed_value = make_hashable(hashable_argument)
                return hashed_value
            return None
            
    return wrapper
# wrap the builtin hash function
hash = hash_decorator(hash)

# save loading times without brittle code
def auto_cache(function, *args, **kwargs):
    # 
    # create hash for arguments
    # 
    try:
        unique_hash = str(function.__name__)+"_"+str(hash(hash((args, kwargs))))
    except:
        unique_hash = None
    if type(unique_hash) != str:
        print(f"the arguments for {function.__name__} couldn't be auto cached")
        print("It probably contains some value that python doesn't know how to hash")
        print('args = ', args)
        print('kwargs = ', kwargs)
        print("running the function manually instead (failsafe)")
        return function(*args, **kwargs)
    
    # make the folders for the cache
    path_to_cache = relative_path("cache.nosync", f"{unique_hash}")
    try:
        os.makedirs(os.path.dirname(path_to_cache))
    except:
        pass
    
    # if the cache (for these arguments) exists, then just load it
    if os.path.exists(path_to_cache):
        return large_pickle_load(path_to_cache)
    # otherwise create it
    else:
        print(f"cache for {function.__name__} (with the current args) didn't exist")
        print("building cache now...")
        result = function(*args, **kwargs)
        try:
            large_pickle_save(result, path_to_cache)
            print("cache built")
        except:
            print(f"the result of {function.__name__} couldn't be auto cached")
            print("It probably contains some value that python doesn't know how to pickle")
            print("or the size of the output is larger than 4gb (less likely)")
            print('args = ', args)
            print('kwargs = ', kwargs)
            print("running the function manually instead (failsafe)")
        return result
