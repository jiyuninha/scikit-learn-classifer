# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/jiyuninha/scikit-learn-classifer.git

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC

def load_dataset(dataset_path):
    # To-Do: Implement this function
    return pd.read_csv(dataset_path)


def dataset_stat(dataset_df):
    # To-Do: Implement this function
    n_feats = dataset_df.shape[1]
    n_class0 = dataset_df[(dataset_df['target'] == 0)].shape[0]
    n_class1 = dataset_df[(dataset_df['target'] == 1)].shape[0]
    return n_feats, n_class0, n_class1


def split_dataset(dataset_df, testset_size):
    # To-Do: Implement this function
    X = dataset_df.drop(['target'], axis=1)
    y = dataset_df['target'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testset_size, random_state=1, shuffle=True,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test


def decision_tree_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    tree = DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=1)
    tree.fit(x_train, y_train)
    test_pred_decision_tree = tree.predict(x_test)
    acc = metrics.accuracy_score(y_test, test_pred_decision_tree)
    prec = metrics.precision_score(y_test, test_pred_decision_tree)
    recall = metrics.recall_score(y_test, test_pred_decision_tree)
    return acc, prec, recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    forest = RandomForestClassifier()
    forest.fit(x_train, y_train)

    test_pred_decision_forest = forest.predict(x_test)
    acc = metrics.accuracy_score(y_test, test_pred_decision_forest)
    prec = metrics.precision_score(y_test, test_pred_decision_forest)
    recall = metrics.recall_score(y_test, test_pred_decision_forest)
    return acc, prec, recall


def svm_train_test(x_train, x_test, y_train, y_test):
    svm_m = SVC()
    svm_m.fit(x_train, y_train)
    test_pred_decision_svm = svm_m.predict(x_test)
    acc = metrics.accuracy_score(y_test, test_pred_decision_svm)
    prec = metrics.precision_score(y_test, test_pred_decision_svm)
    recall = metrics.recall_score(y_test, test_pred_decision_svm)
    return acc, prec, recall

# To-Do: Implement this function

def print_performances(acc, prec, recall):
    # Do not modify this function!
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)


if __name__ == '__main__':
    # Do not modify the main script!
    data_path = sys.argv[1]  #
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print("Number of features: ", n_feats)
    print("Number of class 0 data entries: ", n_class0)
    print("Number of class 1 data entries: ", n_class1)

    print("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print("\nDecision Tree Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print("\nSVM Performances")
    print_performances(acc, prec, recall)