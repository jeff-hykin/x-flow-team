import math
import json
from collections import Counter # frequency count

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
