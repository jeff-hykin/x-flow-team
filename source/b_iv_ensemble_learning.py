from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

def AdaBoost(features, labels):
    base_models = [
        DecisionTreeClassifier(max_depth=1),
        svm.SVC(kernel='linear', C=1),
        MLPClassifier(random_state=1, max_iter=300),
        KNeighborsClassifier(n_neighbors=3)
    ]
    result = []
    # try each base model
    for base_model in base_models:
        clf = AdaBoostClassifier(base_estimator=base_model, n_estimators=100, random_state=0)
        # 5-fold cross validation
        cv_results = cross_validate(clf, features, labels, cv=5)
        result.append(cv_results['test_score'])
    return result

if __name__=="__main__":
    # wrapper features

    # filter features

    # compare the results
    plt.title("compare between wrapper and filter")
    plt.plot([1,2,3,4,5], result1)
    plt.plot([1,2,3,4,5], result2)
    plt.show()
