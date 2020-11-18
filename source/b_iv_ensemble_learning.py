from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

def AdaBoost(features, labels):
    # the base estimator is DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    # 5-fold cross validation
    cv_results = cross_validate(clf, features, labels, cv=5)
    return cv_results['test_score']

if __name__=="__main__":
    # wrapper features

    # filter features

    # compare the results
    plt.title("compare between wrapper and filter")
    plt.plot([1,2,3,4,5], result1)
    plt.plot([1,2,3,4,5], result2)
    plt.show()
