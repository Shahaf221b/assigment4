import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# def split_test_train(dataset):
    # train_set = []
    # test_set = []
    # kf = KFold(n_splits=10 ,shuffle=True, random_state=1)
    # for train, test in kf.split(dataset):
    #     # print("%s %s" % (train, test))
    #     train_set.append(train)
    #     test_set.append(test)
    #     # configure the cross-validation procedure
    #     kf_inner = KFold(n_splits=3, shuffle=True, random_state=1)
    #     # define the model
    #     LR = LogisticRegression(random_state=1)

def classifier(X, y):
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    adaboost_outer_results = list()
    randomforest_outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # -------------- AdaBoost -------------- #
        model_AdaBoost = AdaBoostClassifier(random_state=1)
        # define adaboost_search adaboost_space
        adaboost_space = dict()
        adaboost_space['n_estimators'] = [10, 100, 500]
        adaboost_space['learning_rate'] = [10.0, 5.0, 1.0]
        # define adaboost_search
        adaboost_search = GridSearchCV(model_AdaBoost, adaboost_space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute adaboost_search
        adaboost_result = adaboost_search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        adaboost_best_model = adaboost_result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = adaboost_best_model.predict(X_test)
        # evaluate the model - TODO: more evaluation parameters ?
        acc = accuracy_score(y_test, yhat)
        # store the adaboost_result
        adaboost_outer_results.append(acc)
        # report adaboost progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, adaboost_result.best_score_, adaboost_result.best_params_)) #TODO: remove from print?

        # -------------- RandomForest -------------- #
        model_RandomForest = RandomForestClassifier(random_state=1)
        # define randomforest_search randomforest_space
        randomforest_space = dict()
        randomforest_space['n_estimators'] = [10, 100, 500]
        randomforest_space['max_features'] = [2, 4, 6]
        # define randomforest_search
        randomforest_search = GridSearchCV(model_RandomForest, randomforest_space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute randomforest_search
        randomforest_result = randomforest_search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        randomforest_best_model = randomforest_result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = randomforest_best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        # store the randomforest_result
        randomforest_outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, randomforest_result.best_score_, randomforest_result.best_params_))
    # summarize the estimated performance of the models
    print("--- estimated performance of AdaBoost: ---")
    print('Accuracy: %.3f (%.3f)' % (np.mean(adaboost_outer_results), np.std(adaboost_outer_results))) #TODO: remove from print?
    print("--- estimated performance of RandomForest: ---")
    print('Accuracy: %.3f (%.3f)' % (np.mean(randomforest_outer_results), np.std(randomforest_outer_results)))
# wine = datasets.load_wine()
# wineData = pd.DataFrame(data=np.c_[wine['data'],wine['target']],columns=wine['feature_names']+['target'])
# # wineData.info() # 
# train,test =split_test_train(wineData)
# print(f"train is: {train}")
# print(f"test is: {test}")
print("----------- running models for - wine dataset -----------")
X, y = datasets.load_wine(return_X_y=True)
classifier(X,y)
print("----------- running models for - iris dataset -----------")
X, y = datasets.load_iris(return_X_y=True)
classifier(X,y)
print("----------- running models for - breast_cancer dataset -----------")
X, y = datasets.load_breast_cancer(return_X_y=True)
classifier(X,y)
print("----------- running models for - diabetes dataset -----------")
X, y = datasets.load_diabetes(return_X_y=True)
classifier(X,y)
#TODO: add 6 more datasets

