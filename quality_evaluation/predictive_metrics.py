#!/usr/bin/env python
# coding: utf-8



"""Module with quality metrics for comparison of synthetic datasets with the original dataset from a predictive perspective"""



import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import random
from datetime import date
import datetime as DT
from datetime import datetime
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from collections import Counter
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold


def TB_TOH(model, df_ori, privacy_levels, df_ori_val, dropdummies = None, num_vars = None):
    
    """
    Computes the accuracy difference between an original and synthetic model, both tested on an original holdout set,
    for a given classifier (model).
    Privacy_levels is a list consisting of different privacy levels, multiple dataframes per privacy levels are possible.
    Should be passed as following: [[low1, low2, low3],[medium1, medium2, medium3], [high1, high2, high3]].
    Dropdummies and num_vars are not required arguments, but can be passed in case of Logistic Regression.
    """
         
    print("Applied model is", model.__class__.__name__)

    # Create empty list for the quality measure
    ppds = []
    stds = []
    
    # First create original model
    
    # Drop dummy variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if dropdummies == None:
            print("Warning: no dummy columns specified to drop while regression is applied!")
        else:
            df_ori = df_ori.drop(columns=dropdummies);
            df_ori_val = df_ori_val.drop(columns=dropdummies);
            print("Dummy columns dropped:", dropdummies)

    # Scale numerical variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if num_vars == None:
            print("Warning: no numerical variables specified to scale while regression is applied!")
        else:
            df_ori[num_vars] = preprocessing.minmax_scale(df_ori[num_vars].astype(np.float64))
            df_ori_val[num_vars] = preprocessing.minmax_scale(df_ori_val[num_vars].astype(np.float64))
            print("Numerical variables scaled:", num_vars)
    
    # Set X and y for original data
    X_ori = df_ori.iloc[:,:-1]
    y_ori = df_ori.iloc[:,-1]    
    
    # Set X and y for the holdout validation set of the original data
    X_ori_val = df_ori_val.iloc[:,:-1]
    y_ori_val = df_ori_val.iloc[:,-1]   
    
    # Attributes in original data not present in the validation data due to one hot encoding are added with value 0
    for column in X_ori.columns:
        if column not in X_ori_val.columns:
            print(column, 'present in original data, thus added to validation data with value 0')
            X_ori_val[column] = 0
    
    # Find best models with gridsearchCV
    pipe = Pipeline([('classifier' , model)])

    # Create param grid
    if model.__class__.__name__ == 'LogisticRegression':
        param_grid = [
            {'classifier' : [model],
             'classifier__penalty' : ['l2'],
            'classifier__C' : [0.0001, .001, .01, .1, 1, 10, 100],
            'classifier__solver' : ['lbfgs'],
            'classifier__max_iter' : [4000]}
        ]
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        param_grid = [
            {'classifier' : [model],
             'classifier__criterion' : ['entropy'], # both (also gini) are considered, but after experimentation to speed up the process only entropy is kept
             'classifier__min_samples_leaf' : [0.005], # Avoid overfitting by minimizing the number of samples in leaf node
             'classifier__max_depth' : list(range(1,6))} #  Avoid overfitting by limiting the depth
            ]
    else:
        print("param grid not available for this model")

    # Create grid search object
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5)

    # Fit on data
    best_clf_ori = clf.fit(X_ori, y_ori)
    print("ori:", best_clf_ori.best_estimator_.get_params()['classifier'])
    ori_acc = best_clf_ori.score(X_ori_val, y_ori_val)
    print('Prediction accuracy of original model:', ori_acc)

    for df_syns in privacy_levels:
        # Make copy of df_syn to drop columns not on original dataframe
        df_syns_copy = []
        syn_accs = []
        for df_syn in df_syns:
            df_syn_copy = df_syn.copy()
            df_syns_copy.append(df_syn_copy) 

        # Drop dummy variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if dropdummies == None:
                print("Warning: no dummy columns specified to drop while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    try:
                        df_syn_copy.drop(columns=dropdummies, inplace=True);
                        print("Dummy columns dropped:", dropdummies)
                    except KeyError: 
                        try:
                            dropdummies2 = ['stage_2', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies2, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)
                        except KeyError:
                            dropdummies3 = ['stage_4', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies3, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)

        # Scale numerical variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if num_vars == None:
                print("Warning: no numerical variables specified to scale while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    df_syn_copy[num_vars] = preprocessing.minmax_scale(df_syn_copy[num_vars].astype(np.float64))
                print("Numerical variables scaled:", num_vars)

        # Fit for every synthetic dataset, and compare with the original model
        for df_syn_copy in df_syns_copy:
            # Set X and y for synthetic data
            X_syn = df_syn_copy.iloc[:,:-1]
            y_syn = df_syn_copy.iloc[:,-1]
            
            try:
                best_clf_syn = clf.fit(X_syn, y_syn)
                print("syn:", best_clf_syn.best_estimator_.get_params()['classifier'])
        
                # Set X and y for the holdout validation set of the original data again, in case columns were dropped before in the next line of this loop
                X_ori_val = df_ori_val.iloc[:,:-1]
                y_ori_val = df_ori_val.iloc[:,-1]
                
                # Attributes in original data not present in the validation data due to one hot encoding are added with value 0 to validation set
                for column in X_ori.columns:
                    if column not in X_ori_val.columns:
                        print(column, 'present in original data, thus added to validation data with value 0')
                        X_ori_val[column] = 0

                # If attribute value due to one hot encoding not present in synthetic data as attribute, drop in validation set
                for column in X_ori_val.columns:
                    if column not in X_syn.columns:
                        print(column, 'not present in synthetic dataset, thus dropped from validation set')
                        X_ori_val.drop([column], axis=1, inplace=True)

                syn_acc = best_clf_syn.score(X_ori_val, y_ori_val)
                print('Prediction accuracy of synthetic model:', syn_acc)

                # Add syn_acc to list of all syn_accs for this epsilon level
                syn_accs.append(syn_acc)
            except ValueError:
                syn_accs.append(0) # If there is only one class left in the synthetic dataset, the quality is considered as bad, thus a 0 is added
                print("Only one class left in synthetic dataset, no Logistic Regression possible. Prediction accuracy is set to 0")
                
        # Calculate Predictive Power Difference
        ppd = ori_acc - np.mean(syn_accs)
        std = np.std(syn_accs)
        print("TB-TOH Accuracy Difference:", ppd, "with standard deviation:", std)
        ppds.append(ppd)
        stds.append(std)

    return ori_acc, ppds, stds



def TB_TOHmodels(models, df_ori, privacy_levels, df_ori_val, dropdummies=None, num_vars=None):
    
    """
    Computes the TB_TOH accuracy difference for multiple models. 
    Models should be given as a list.
    Returns the accuracies of the original models, the accuracy differences and the standard deviation per privacy level.
    """
    
    original_accs = []
    result = []
    errorbars = []
    
    for model in models:
        ori_accs, ppdss, stdss = TB_TOH(model, df_ori, privacy_levels, df_ori_val, dropdummies, num_vars);
        print("TB_TOH Accuracay Differences for", model.__class__.__name__, ":", ppdss)
        
        result.append(model.__class__.__name__)
        result.append(ppdss)
        errorbars.append(model.__class__.__name__)
        errorbars.append(stdss)
        original_accs.append(model.__class__.__name__)
        original_accs.append(ori_accs)
    
    return original_accs, result, errorbars



def TO_TB(model, df_ori, privacy_levels, df_ori_val, dropdummies = None, num_vars = None):    

    """
    Computes the accuracy difference between an original and synthetic dataset, both tested on an original model,
    for a given classifier (model).
    It also returns the support distance in case of decision tree.
    Privacy_levels is a list consisting of different privacy levels, multiple dataframes per privacy levels are possible.
    Should be passed as following: [[low1, low2, low3],[medium1, medium2, medium3], [high1, high2, high3]].
    Dropdummies and num_vars are not required arguments, but can be passed in case of Logistic Regression.
    """

    print("Applied model is", model.__class__.__name__)
    
    # Create empty lists for the quality measures
    ras = []
    stds = []
    rsds = []
    stds2 = []
    
    # First create original model
    
    # Drop dummy variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if dropdummies == None:
            print("Warning: no dummy columns specified to drop while regression is applied!")
        else:
            df_ori = df_ori.drop(columns=dropdummies);
            df_ori_val = df_ori_val.drop(columns=dropdummies);
            print("Dummy columns dropped:", dropdummies)

    # Scale numerical variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if num_vars == None:
            print("Warning: no numerical variables specified to scale while regression is applied!")
        else:
            df_ori[num_vars] = preprocessing.minmax_scale(df_ori[num_vars].astype(np.float64))
            df_ori_val[num_vars] = preprocessing.minmax_scale(df_ori_val[num_vars].astype(np.float64))
            print("Numerical variables scaled:", num_vars)
    
        # Set X and y for original data
    X_ori = df_ori.iloc[:,:-1]
    y_ori = df_ori.iloc[:,-1]

    # Find best models with gridsearchCV
    pipe = Pipeline([('classifier' , model)])

    # Create param grid
    if model.__class__.__name__ == 'LogisticRegression':
        param_grid = [
            {'classifier' : [model],
             'classifier__penalty' : ['l2'],
             'classifier__C' : [0.0001, .001, .01, .1, 1, 10, 100],
             'classifier__solver' : ['lbfgs'],
             'classifier__max_iter' : [4000]}
        ]
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
         param_grid = [
            {'classifier' : [model],
             'classifier__criterion' : ['entropy'],
             'classifier__min_samples_leaf' : [0.005], # Avoid overfitting by minimizing the number of samples in leaf node
             'classifier__max_depth' : list(range(1,6))} #  Avoid overfitting by limiting the depth
        ]
    else:
        print("param grid not available for this model")

    # Create grid search object, avoid overfitting by applying cross validation and restrictions in paramgrid
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5)

    # Fit on data
    best_clf_ori = clf.fit(X_ori, y_ori)
    best_params = clf.best_params_

    # Accuracy of original training data with which the model was also refitted
    ori_acc = best_clf_ori.score(X_ori, y_ori)
    print('Accuracy on data that trained the model:', ori_acc)
    
    for df_syns in privacy_levels:
        # Make copy of df_syn to drop columns not on original dataframe
        df_syns_copy = []
        syn_accs = []
        syn_rsd = []
        for df_syn in df_syns:
            df_syn_copy = df_syn.copy()
            df_syns_copy.append(df_syn_copy) 

        # Drop dummy variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if dropdummies == None:
                print("Warning: no dummy columns specified to drop while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    try:
                        df_syn_copy.drop(columns=dropdummies, inplace=True);
                        print("Dummy columns dropped:", dropdummies)
                    except KeyError: 
                        try:
                            dropdummies2 = ['stage_2', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies2, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)
                        except KeyError:
                            dropdummies3 = ['stage_4', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies3, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)

        # Scale numerical variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if num_vars == None:
                print("Warning: no numerical variables specified to scale while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    df_syn_copy[num_vars] = preprocessing.minmax_scale(df_syn_copy[num_vars].astype(np.float64))
                print("Numerical variables scaled:", num_vars)
        
        # Test with every synthetic dataset, and compare with the original dataset
        for df_syn_copy in df_syns_copy:
            # Set X and y for synthetic data
            X_syn = df_syn_copy.iloc[:,:-1]
            y_syn = df_syn_copy.iloc[:,-1]

            # Attributes in original data not present in the synthetic data due to one hot encoding are added with value 0
            for column in X_ori.columns:
                if column not in X_syn.columns:
                    print(column, 'present in original data, thus added to synthetic data with value 0')
                    X_syn[column] = 0

            # Accuracy of synthetic data on this best estimator that is fit on all the original data
            syn_acc = best_clf_ori.score(X_syn, y_syn)
            print('Prediction accuracy of synthetic data:', syn_acc)
            
            # Add syn_acc to list of all syn_accs for this epsilon level
            syn_accs.append(syn_acc)
            
            if model.__class__.__name__ == 'DecisionTreeClassifier':
                rsd = SupportDistance(df_ori, df_syn_copy, df_ori_val, best_params)
                syn_rsd.append(rsd)
            #else:
            #    print("Rule Support Distance not applicable")

        # Compute Rule Accuracy
        RA = ori_acc - np.mean(syn_accs)
        std = np.std(syn_accs)
        print('TO-TB accuracy difference with same training data:', RA, "with standard deviation:", std)
        ras.append(RA)
        stds.append(std)
        
        # Compute Support Distance
        if model.__class__.__name__ == 'DecisionTreeClassifier':
            RSD = np.mean(syn_rsd)
            std2 = np.std(syn_rsd)
            rsds.append(RSD)
            stds2.append(std2)
        else:
            print("Support Distance not applicable")

    return ori_acc, ras, stds, rsds, stds2



def SupportDistance(df_ori, df_syn, df_ori_val, best_params):
    
    """
    Computes the support distance (if a decision tree) between an original dataset and synthetic dataset.
    Same parameters are used as the best model that was found for TO_TB.
    Else, enter own best_params.
    """

    X_ori = df_ori.iloc[:,:-1]
    y_ori = df_ori.iloc[:,-1]

    # Set X and y for synthetic data
    X_syn = df_syn.iloc[:,:-1]
    y_syn = df_syn.iloc[:,-1]
    
    # Attributes in original data not present in the synthetic data due to one hot encoding are added with value 0
    for column in X_ori.columns:
        if column not in X_syn.columns:
            print(column, 'present in original data, thus added to synthetic data with value 0')
            X_syn[column] = 0

    # Set X and y for the holdout validation set of the original data
    X_ori_val = df_ori_val.iloc[:,:-1]
    y_ori_val = df_ori_val.iloc[:,-1]
    
    model = best_params.get("classifier")
    dt = model.fit(X_ori, y_ori)

    # Return the index of the leaf that each sample is predicted as
    ori_leaf_preds = dt.apply(X_ori) # Or X_ori_val, we know all rules are used by the original data if we use X_ori instead of X_ori_val

    # Count the number of samples per leaf node 
    ori_key_leaf = Counter(ori_leaf_preds).keys() 
    ori_samples_leaf = Counter(ori_leaf_preds).values() 
    # Order both lists low to high
    ori_key_leaf, ori_samples_leaf = (list(t) for t in zip(*sorted(zip(ori_key_leaf, ori_samples_leaf))))
    
    # Same for synthetic data
    syn_leaf_preds = dt.apply(X_syn)

    # Count the number of samples per leaf node
    syn_key_leaf = Counter(syn_leaf_preds).keys() 
    syn_samples_leaf = Counter(syn_leaf_preds).values()
    # Order both lists low to high
    syn_key_leaf, syn_samples_leaf = (list(t) for t in zip(*sorted(zip(syn_key_leaf, syn_samples_leaf))))
    
    # Compute the support pe rule by dividing over the total number of samples
    syn_support = [i/sum(syn_samples_leaf) for i in syn_samples_leaf]
    ori_support = [i/sum(ori_samples_leaf) for i in ori_samples_leaf]
    
    print('total number of samples is:', sum(syn_samples_leaf))
    print('total number of ori rules is:', len(ori_key_leaf))
    print('total number of rules that syn follows from ori are:', len(syn_key_leaf))
      
    # Make dictionary of key and support lists
    ori_dict = dict(zip(ori_key_leaf, ori_support))
    syn_dict = dict(zip(syn_key_leaf, syn_support))

    # Important check to see if both contain all the leaf nodes for when the lists are compared on index
    print('Both datasets use the same leaf nodes:', set(ori_dict.keys()) == set(syn_dict.keys()))

    # Sum up the rule support difference per rule (leaf node)
    sum_sup_dif = 0
    for key in ori_dict.keys():
        if key in syn_dict.keys():
            sup_dif = abs(ori_dict[key] - syn_dict[key])
            sum_sup_dif += sup_dif
        else:
            print("leaf node not reached by synthetic dataset")
            sum_sup_dif += ori_dict[key]

    # Compute the Distance Support
    RSD = sum_sup_dif/len(ori_key_leaf) # Divide total support difference by total number of orignal rules (leaf nodes)
    print('Support distance:', RSD)
    
    return RSD



def TO_TBmodels(models, df_ori, privacy_levels, df_ori_val, dropdummies=None, num_vars=None):
    
    """
    Computes the TO_TB accuracy difference for multiple models. 
    Models should be given as a list.
    Returns the accuracies of the original models, the accuracy differences and the standard deviation per privacy level.
    Returns the support distance and the standard deviation per privacy level, if applicable (decision tree).
    """
    
    original_accs = []
    resultRA = []
    errorbarsRA = []
    resultRSD = []
    errorbarsRSD = []
    
    for model in models:
        ori_accs, rass, stdss, rsdss, stdss2 = TO_TB(model, df_ori, privacy_levels, df_ori_val, dropdummies, num_vars);
        print("TB-TO accuracy difference and support difference for", model.__class__.__name__, ":", rass, rsdss)
        original_accs.append(model.__class__.__name__)
        original_accs.append(ori_accs)
        resultRA.append(model.__class__.__name__)
        resultRA.append(rass) 
        errorbarsRA.append(model.__class__.__name__)
        errorbarsRA.append(stdss)
        resultRSD.append(model.__class__.__name__)
        resultRSD.append(rsdss)
        errorbarsRSD.append(model.__class__.__name__)
        errorbarsRSD.append(stdss2)
    
    return original_accs, resultRA, errorbarsRA, resultRSD, errorbarsRSD



def FIDbenchmarking(models, df_ori, dropdummies = None, num_vars = None):
    
    """
    Benchmark which classifier is stable over different folds of the dataset,
    by looking at the mean accuracy score and standard deviation of this classifier over the folds for the original dataset.
    Not automated yet, output should be checked by hand to see if classifier is stable.
    """
    
    # Set X and y for original data
    X_ori = df_ori.iloc[:,:-1]
    y_ori = df_ori.iloc[:,-1]

    # Split data up into 5 folds to execute gridsearch on 5 different folds to check stability of classifiers

    # Split into 5 stratified folds, so 5 "different" original sets with corresponding holdout validation sets are created
    skf = StratifiedKFold(n_splits=5, random_state=8, shuffle=True)

    # Save indices per split so separate dataframes can be saved later
    val_idx = []
    for train_index, test_index in skf.split(X_ori, y_ori):
        val_idx.append(test_index)

    # Create the 5 folds by saving the 5 validation sets
    fold_1 = df_ori.loc[val_idx[0]]
    fold_2 = df_ori.loc[val_idx[1]]
    fold_3 = df_ori.loc[val_idx[2]]
    fold_4 = df_ori.loc[val_idx[3]]
    fold_5 = df_ori.loc[val_idx[4]]

    folds = [fold_1, fold_2, fold_3, fold_4, fold_5]

    i=1
    DTscores = []
    LRscores = []
    for fold in folds:
        print("fold", i)
        DTscores.extend(("fold", i))
        LRscores.extend(('fold', i))

        for model in models:
            pipe = Pipeline([('classifier' , model)])

            # Drop dummy variables in case of regression
            fold_copy = fold.copy()
            if model.__class__.__name__ == 'LogisticRegression':
                if dropdummies == None:
                    print("Warning: no dummy columns specified to drop while regression is applied!")
                else:
                    df_ori_copy = fold_copy.drop(columns=dropdummies);
                    print("Dummy columns dropped:", dropdummies)

            # Scale numerical variables in case of regression
            if model.__class__.__name__ == 'LogisticRegression':
                if num_vars == None:
                    print("Warning: no numerical variables specified to scale while regression is applied!")
                else:
                    fold_copy[num_vars] = preprocessing.minmax_scale(fold_copy[num_vars].astype(np.float64))
                    print("Numerical variables scaled:", num_vars)

            # Set X and y for the fold for this model
            X = fold_copy.iloc[:,:-1]
            y = fold_copy.iloc[:,-1]

            # Create param grid
            if model.__class__.__name__ == 'LogisticRegression':
                param_grid = [
                    {'classifier' : [model],
                     'classifier__penalty' : ['l2'],
                    'classifier__C' : [0.0001, .001, .01, .1, 1, 10, 100],
                    'classifier__solver' : ['lbfgs'],
                    'classifier__max_iter' : [4000]}
                ]
            elif model.__class__.__name__ == 'DecisionTreeClassifier':
                param_grid = [
                    {'classifier' : [model],
                     'classifier__criterion' : ['entropy'], # both (also gini) are considered, but after experimentation to speed up the process only entropy is kept
                     'classifier__min_samples_leaf' : [0.005], # Avoid overfitting by minimizing the number of samples in leaf node
                     'classifier__max_depth' : list(range(1,6))} #  Avoid overfitting by limiting the depth
                    ]
            else:
                print("param grid not available for this model")

            # Create grid search object with random state for cv folds so DT and LR are tested on same folds
            clf = GridSearchCV(pipe, param_grid = param_grid, cv = StratifiedKFold(5, shuffle=True, random_state=2))

            print("model", model)

            # Fit on data
            best_clf_ori = clf.fit(X, y)

            best_mean = best_clf_ori.cv_results_['mean_test_score'][best_clf_ori.best_index_] 
            best_std = best_clf_ori.cv_results_['std_test_score'][best_clf_ori.best_index_] 

            print("best parameters:", best_clf_ori.best_estimator_.get_params()['classifier'])
            print("mean", best_mean)
            print("std", best_std)
            
            if model.__class__.__name__ == 'LogisticRegression':
                LRscores.extend(("mean", best_mean, "std", best_std))
            elif model.__class__.__name__ == 'DecisionTreeClassifier':    
                DTscores.extend(("mean", best_mean, "std", best_std))
        i+=1

    return LRscores, DTscores



def FID(model, df_ori, privacy_levels, df_ori_val, dropdummies = None, num_vars = None):    

    """
    Computes the feature importance difference for a given algorithm, original dataset and synthetic dataset.
    Dropdummies and numerical variables are not a required argument, but can be passed in case of Logistic Regression.
    """
    
    print("Applied model is", model.__class__.__name__)

    # Create empty list for the quality measure
    fids = []
    stds = []
    
    # First create original model
    
    # Drop dummy variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if dropdummies == None:
            print("Warning: no dummy columns specified to drop while regression is applied!")
        else:
            df_ori = df_ori.drop(columns=dropdummies);
            df_ori_val = df_ori_val.drop(columns=dropdummies);
            print("Dummy columns dropped:", dropdummies)

    # Scale numerical variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if num_vars == None:
            print("Warning: no numerical variables specified to scale while regression is applied!")
        else:
            df_ori[num_vars] = preprocessing.minmax_scale(df_ori[num_vars].astype(np.float64))
            df_ori_val[num_vars] = preprocessing.minmax_scale(df_ori_val[num_vars].astype(np.float64))
            print("Numerical variables scaled:", num_vars)
    
    # Set X and y for original data
    X_ori = df_ori.iloc[:,:-1]
    y_ori = df_ori.iloc[:,-1]
    
    # Set X and y for the holdout validation set of the original data
    X_ori_val = df_ori_val.iloc[:,:-1]
    y_ori_val = df_ori_val.iloc[:,-1]   
    
    # Attributes in original data not present in the validation data due to one hot encoding are added with value 0
    for column in X_ori.columns:
        if column not in X_ori_val.columns:
            print(column, 'present in original data, thus added to validation data with value 0')
            X_ori_val[column] = 0
    
    # Find best model with gridsearchCV
    pipe = Pipeline([('classifier' , model)])

    # Create param grid
    if model.__class__.__name__ == 'LogisticRegression':
        param_grid = [
            {'classifier' : [model],
             'classifier__penalty' : ['l2'],
            'classifier__C' : [0.0001, .001, .01, .1, 1, 10, 100],
            'classifier__solver' : ['lbfgs'],
            'classifier__max_iter' : [4000]}
        ]
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        param_grid = [
            {'classifier' : [model],
             'classifier__criterion' : ['entropy'], # both (also gini) are considered, but after experimentation to speed up the process only entropy is kept
             'classifier__min_samples_leaf' : [0.005], # Avoid overfitting by minimizing the number of samples in leaf node
             'classifier__max_depth' : list(range(1,6))} #  Avoid overfitting by limiting the depth
            ]
    else:
        print("param grid not available for this model")

    # Create grid search object
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
    
    # Fit on data
    best_clf_ori = clf.fit(X_ori, y_ori)
    best_params_ori = clf.best_params_
    print("ori:", best_clf_ori.best_estimator_.get_params()['classifier'])

    # Create model with best parameters and fit
    model_ori = best_params_ori.get("classifier")
    fitted_ori = model_ori.fit(X_ori, y_ori)
    
    # Save original accuracy for minimal accuracy requirement
    ori_acc = fitted_ori.score(X_ori_val, y_ori_val)
    print('accuracy of original model is', ori_acc)

    # Different ways to extract the feature importances for Decision Tree and Logistic Regression
    if model.__class__.__name__ == 'DecisionTreeClassifier':
        ori_fi = fitted_ori.feature_importances_
    elif model.__class__.__name__ == 'LogisticRegression':
        ori_fi = (np.std(X_ori, 0)*abs(fitted_ori.coef_[0]))
    else:
        print("feature importances not specified for this model")

    tot=0
    for i in range(0,len(ori_fi)):
        print('Feature:', X_ori.columns[i],'Score: %.5f' % (ori_fi[i]))
        tot+=ori_fi[i]
    print('total is', tot)

    # Plot feature importances
    pyplot.bar([x for x in range(len(ori_fi))], ori_fi)
    pyplot.show()

    # Now create synthetic models
    
    for df_syns in privacy_levels:
        # Make copy of df_syn to drop columns not on original dataframe
        df_syns_copy = []
        syn_fids = []
        for df_syn in df_syns:
            df_syn_copy = df_syn.copy()
            df_syns_copy.append(df_syn_copy) 

        # Drop dummy variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if dropdummies == None:
                print("Warning: no dummy columns specified to drop while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    try:
                        df_syn_copy.drop(columns=dropdummies, inplace=True);
                        print("Dummy columns dropped:", dropdummies)
                    except KeyError: 
                        try:
                            dropdummies2 = ['stage_2', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies2, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)
                        except KeyError:
                            dropdummies3 = ['stage_4', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies3, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)

        # Scale numerical variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if num_vars == None:
                print("Warning: no numerical variables specified to scale while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    df_syn_copy[num_vars] = preprocessing.minmax_scale(df_syn_copy[num_vars].astype(np.float64))
                print("Numerical variables scaled:", num_vars)
            
        # Fit for every synthetic dataset, and compare with the original model
        for df_syn_copy in df_syns_copy:
            # Set X and y for synthetic data
            X_syn = df_syn_copy.iloc[:,:-1]
            y_syn = df_syn_copy.iloc[:,-1]

            # Attributes in original data not present in the synthetic data due to one hot encoding are added with value 0
            for column in X_ori.columns:
                if column not in X_syn.columns:
                    print(column, 'present in original data, thus added to synthetic data with value 0')
                    # Add at original location of original dataset since same order is needed for computing the FID per feature
                    index = X_ori.columns.get_loc(column)
                    X_syn.insert(index, column, ([0]*len(X_syn)))

            try:
                best_clf_syn = clf.fit(X_syn, y_syn)
                best_params_syn = clf.best_params_
                print("syn:", best_clf_syn.best_estimator_.get_params()['classifier'])

                # Create model with best parameters and fit
                model_syn = best_params_syn.get("classifier")
                fitted_syn = model_syn.fit(X_syn, y_syn)

                # Different ways to extract the feature importances for Decision Tree and Logistic Regression
                if model.__class__.__name__ == 'DecisionTreeClassifier':
                    syn_fi = fitted_syn.feature_importances_
                elif model.__class__.__name__ == 'LogisticRegression':
                    syn_fi = (np.std(X_syn, 0)*abs(fitted_syn.coef_[0]))
                else:
                    print("feature importances not specified for this model")

                tot=0
                for i in range(0,len(syn_fi)):
                    print('Feature:', X_syn.columns[i],'Score: %.5f' % (syn_fi[i]))
                    tot+=syn_fi[i]
                print('total is', tot)

                # Plot feature importances
                pyplot.bar([x for x in range(len(syn_fi))], syn_fi)
                pyplot.show()
            except ValueError:
                syn_fi = [0]*len(ori_fi) # If there is only one class left in the synthetic dataset, LR is not possible and all coef are set to 0 to represent bad quality
                print("Only one class left in synthetic dataset, no Logistic Regression possible. Coefficients are set to 0")
                
            # Sum up the feature importance difference per attribute
            sum_fi_dif = 0
            for i in range(0,len(ori_fi)):
                fi_dif = (ori_fi[i] - syn_fi[i])**2 #This is for RMSE. MAE: abs(ori_fi[i] - syn_fi[i])
                print('Attribute', X_ori.columns[i], 'has RMSE feature importance difference', fi_dif)
                sum_fi_dif += fi_dif

            # Compute the Feature Importance Difference
            fid = sum_fi_dif/len(ori_fi) # Divide total feature importance difference by total number of attributes
            fid = math.sqrt(fid) # This is for RMSE, delete this for MAE
            print('Feature importance difference:', fid)

            syn_fids.append(fid)

        # Calculate Feature Importance Difference
        fid_mean = np.mean(syn_fids)
        std = np.std(syn_fids)
        print('Mean Feature importance difference:', fid_mean, "with standard deviation:", std)
        fids.append(fid_mean)
        stds.append(std)

    return ori_acc, fids, stds




def FIDmodels(models, df_ori, privacy_levels, df_ori_val, dropdummies=None, num_vars=None):
    
    """
    Computes the feature importance difference for multiple models. 
    Models should be given as a list.
    Returns the accuracies of the original models, the feature importance differences and the standard deviation per privacy level.
    """
    
    original_accs = []
    result = []
    errorbars = []
    
    for model in models:
        ori_accs, fidss, stdss = FID(model, df_ori, privacy_levels, df_ori_val, dropdummies, num_vars);
        print("Feature Importance Differences for", model.__class__.__name__, ":", fidss)
        
        result.append(model.__class__.__name__)
        result.append(fidss)
        errorbars.append(model.__class__.__name__)
        errorbars.append(stdss)
        original_accs.append(model.__class__.__name__)
        original_accs.append(ori_accs)
    
    return original_accs, result, errorbars


def NormalizedFID(model, df_ori, privacy_levels, df_ori_val, dropdummies = None, num_vars = None):    

    """
    Computes the feature importance difference for a given algorithm, original dataset and synthetic dataset.
    Dropdummies and numerical variables are not a required argument, but can be passed in case of Logistic Regression.
    """
    
    print("Applied model is", model.__class__.__name__)

    # Create empty list for the quality measure
    fids = []
    stds = []
    
    # First create original model
    
    # Drop dummy variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if dropdummies == None:
            print("Warning: no dummy columns specified to drop while regression is applied!")
        else:
            df_ori = df_ori.drop(columns=dropdummies);
            df_ori_val = df_ori_val.drop(columns=dropdummies);
            print("Dummy columns dropped:", dropdummies)

    # Scale numerical variables in case of regression
    if model.__class__.__name__ == 'LogisticRegression':
        if num_vars == None:
            print("Warning: no numerical variables specified to scale while regression is applied!")
        else:
            df_ori[num_vars] = preprocessing.minmax_scale(df_ori[num_vars].astype(np.float64))
            df_ori_val[num_vars] = preprocessing.minmax_scale(df_ori_val[num_vars].astype(np.float64))
            print("Numerical variables scaled:", num_vars)
    
    # Set X and y for original data
    X_ori = df_ori.iloc[:,:-1]
    y_ori = df_ori.iloc[:,-1]
    
    # Set X and y for the holdout validation set of the original data
    X_ori_val = df_ori_val.iloc[:,:-1]
    y_ori_val = df_ori_val.iloc[:,-1]   
    
    # Attributes in original data not present in the validation data due to one hot encoding are added with value 0
    for column in X_ori.columns:
        if column not in X_ori_val.columns:
            print(column, 'present in original data, thus added to validation data with value 0')
            X_ori_val[column] = 0
    
    # Find best model with gridsearchCV
    pipe = Pipeline([('classifier' , model)])

    # Create param grid
    if model.__class__.__name__ == 'LogisticRegression':
        param_grid = [
            {'classifier' : [model],
             'classifier__penalty' : ['l2'],
            'classifier__C' : [0.0001, .001, .01, .1, 1, 10, 100],
            'classifier__solver' : ['lbfgs'],
            'classifier__max_iter' : [4000]}
        ]
    elif model.__class__.__name__ == 'DecisionTreeClassifier':
        param_grid = [
            {'classifier' : [model],
             'classifier__criterion' : ['entropy'], # both (also gini) are considered, but after experimentation to speed up the process only entropy is kept
             'classifier__min_samples_leaf' : [0.005], # Avoid overfitting by minimizing the number of samples in leaf node
             'classifier__max_depth' : list(range(1,6))} #  Avoid overfitting by limiting the depth
            ]
    else:
        print("param grid not available for this model")

    # Create grid search object
    clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
    
    # Fit on data
    best_clf_ori = clf.fit(X_ori, y_ori)
    best_params_ori = clf.best_params_
    print("ori:", best_clf_ori.best_estimator_.get_params()['classifier'])

    # Create model with best parameters and fit
    model_ori = best_params_ori.get("classifier")
    fitted_ori = model_ori.fit(X_ori, y_ori)
    
    # Save original accuracy for minimal accuracy requirement
    ori_acc = fitted_ori.score(X_ori_val, y_ori_val)
    print('accuracy of original model is', ori_acc)

    # Different ways to extract the feature importances for Decision Tree and Logistic Regression
    if model.__class__.__name__ == 'DecisionTreeClassifier':
        ori_fi = fitted_ori.feature_importances_
    elif model.__class__.__name__ == 'LogisticRegression':
        ori_fi = (np.std(X_ori, 0)*abs(fitted_ori.coef_[0]))
    else:
        print("feature importances not specified for this model")

    tot=0
    for i in range(0,len(ori_fi)):
        print('Feature:', X_ori.columns[i],'Score: %.5f' % (ori_fi[i]))
        tot+=ori_fi[i]
    print('total is', tot)
    
    normalized_coeffs=[]
    for i in range(0,len(ori_fi)):
        normalized_coeff = ori_fi[i]/tot
        print('Feature:', X_ori.columns[i],'Normalized score: %.5f' % (normalized_coeff))
        normalized_coeffs.append(normalized_coeff)

    # Plot feature importances
    pyplot.bar([x for x in range(len(ori_fi))], ori_fi)
    pyplot.show()

    # Now create synthetic models
    
    for df_syns in privacy_levels:
        # Make copy of df_syn to drop columns not on original dataframe
        df_syns_copy = []
        syn_fids = []
        for df_syn in df_syns:
            df_syn_copy = df_syn.copy()
            df_syns_copy.append(df_syn_copy) 

        # Drop dummy variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if dropdummies == None:
                print("Warning: no dummy columns specified to drop while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    try:
                        df_syn_copy.drop(columns=dropdummies, inplace=True);
                        print("Dummy columns dropped:", dropdummies)
                    except KeyError: 
                        try:
                            dropdummies2 = ['stage_2', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies2, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)
                        except KeyError:
                            dropdummies3 = ['stage_4', 'subloc_7', "diagnosis_age_71-90", "lymph_pos_0"] # Different columns have to be dropped in case not present in synthetic dataset
                            df_syn_copy.drop(columns=dropdummies3, inplace=True);
                            print("Dummy columns dropped after exception:", dropdummies)

        # Scale numerical variables in case of regression
        if model.__class__.__name__ == 'LogisticRegression':
            if num_vars == None:
                print("Warning: no numerical variables specified to scale while regression is applied!")
            else:
                for df_syn_copy in df_syns_copy:
                    df_syn_copy[num_vars] = preprocessing.minmax_scale(df_syn_copy[num_vars].astype(np.float64))
                print("Numerical variables scaled:", num_vars)
            
        # Fit for every synthetic dataset, and compare with the original model
        for df_syn_copy in df_syns_copy:
            # Set X and y for synthetic data
            X_syn = df_syn_copy.iloc[:,:-1]
            y_syn = df_syn_copy.iloc[:,-1]

            # Attributes in original data not present in the synthetic data due to one hot encoding are added with value 0
            for column in X_ori.columns:
                if column not in X_syn.columns:
                    print(column, 'present in original data, thus added to synthetic data with value 0')
                    # Add at original location of original dataset since same order is needed for computing the FID per feature
                    index = X_ori.columns.get_loc(column)
                    X_syn.insert(index, column, ([0]*len(X_syn)))

            try:
                best_clf_syn = clf.fit(X_syn, y_syn)
                best_params_syn = clf.best_params_
                print("syn:", best_clf_syn.best_estimator_.get_params()['classifier'])

                # Create model with best parameters and fit
                model_syn = best_params_syn.get("classifier")
                fitted_syn = model_syn.fit(X_syn, y_syn)

                # Different ways to extract the feature importances for Decision Tree and Logistic Regression
                if model.__class__.__name__ == 'DecisionTreeClassifier':
                    syn_fi = fitted_syn.feature_importances_
                elif model.__class__.__name__ == 'LogisticRegression':
                    syn_fi = (np.std(X_syn, 0)*abs(fitted_syn.coef_[0]))
                else:
                    print("feature importances not specified for this model")

                tot=0
                for i in range(0,len(syn_fi)):
                    print('Feature:', X_syn.columns[i],'Score: %.5f' % (syn_fi[i]))
                    tot+=syn_fi[i]
                print('total is', tot)
                
                if tot == 0:
                    tot=1
                
                syn_normalized_coeffs=[]
                for i in range(0,len(syn_fi)):
                    syn_normalized_coeff = syn_fi[i]/tot
                    print('Feature:', X_syn.columns[i],'Normalized score: %.5f' % (syn_normalized_coeff))
                    syn_normalized_coeffs.append(syn_normalized_coeff)

                # Plot feature importances
                pyplot.bar([x for x in range(len(syn_fi))], syn_fi)
                pyplot.show()
            except ValueError:
                syn_fi = [0]*len(ori_fi) # If there is only one class left in the synthetic dataset, LR is not possible and all coef are set to 0 to represent bad quality
                print("Only one class left in synthetic dataset, no Logistic Regression possible. Coefficients are set to 0")
                
            # Sum up the feature importance difference per attribute
        #    sum_fi_dif = 0
        #    for i in range(0,len(ori_fi)):
        #        fi_dif = (ori_fi[i] - syn_fi[i])**2 #This is for RMSE. MAE: abs(ori_fi[i] - syn_fi[i])
        #        print('Attribute', X_ori.columns[i], 'has RMSE feature importance difference', fi_dif)
        #        sum_fi_dif += fi_dif
            
            # Sum up the normalized coefficients difference per feature
            sum_fi_dif = 0
            for i in range(0,len(normalized_coeffs)):
                fi_dif = (normalized_coeffs[i] - syn_normalized_coeffs[i])**2 #This is for RMSE. MAE: abs(ori_fi[i] - syn_fi[i])
                print('Attribute', X_ori.columns[i], 'has squared feature importance difference', fi_dif)
                sum_fi_dif += fi_dif

            # Compute the Feature Importance Difference
            fid = sum_fi_dif/len(normalized_coeffs) # Divide total feature importance difference by total number of attributes
            fid = math.sqrt(fid) # This is for RMSE, delete this for MAE
            print('Feature importance difference:', fid)

            syn_fids.append(fid)

        # Calculate Feature Importance Difference
        fid_mean = np.mean(syn_fids)
        std = np.std(syn_fids)
        print('Mean Feature importance difference:', fid_mean, "with standard deviation:", std)
        fids.append(fid_mean)
        stds.append(std)

    return ori_acc, fids, stds



def NormalizedFIDmodels(models, df_ori, privacy_levels, df_ori_val, dropdummies=None, num_vars=None):
    
    """
    Computes the feature importance difference for multiple models. 
    Models should be given as a list.
    Returns the accuracies of the original models, the feature importance differences and the standard deviation per privacy level.
    """
    
    original_accs = []
    result = []
    errorbars = []
    
    for model in models:
        ori_accs, fidss, stdss = NormalizedFID(model, df_ori, privacy_levels, df_ori_val, dropdummies, num_vars);
        print("Feature Importance Differences for", model.__class__.__name__, ":", fidss)
        
        result.append(model.__class__.__name__)
        result.append(fidss)
        errorbars.append(model.__class__.__name__)
        errorbars.append(stdss)
        original_accs.append(model.__class__.__name__)
        original_accs.append(ori_accs)
    
    return original_accs, result, errorbars





# Visualizations



def QMbar_one_DP_alg(QM, QMerror, QMname, labels):
    
    """
    Create bar chart of the quality measure for synthetic datasets over privacy levels for one synthetic DP algorithm.
    Input is similar to output of the quality measures.
    QM is the result.
    QM error is the standard deviation.
    labels is de labels for the privacy levels, in the same order as the synthetic dataframes privacy levels are passed.
    """
    
    data = QM[1::2]
    errors = QMerror[1::2]
    color_list = ['seagreen', 'mediumseagreen', 'sandybrown' , 'orange', 'goldenrod']
    name=0
    
    if data[0] == []:
        data=data[1:]
        errors=errors[1:]
        color_list = ['mediumseagreen', 'sandybrown' , 'orange', 'goldenrod']
        name=2
    
    plt.figure(figsize=(9,6))

    gap = .8 / len(data)
    for i, row in enumerate(data):
        X = np.arange(len(row))
        plt.bar(X + i * gap, row,
                yerr = errors[i],
            align='center',
           capsize=5,
        width = gap,  label=QM[name],
        color = color_list[i % len(color_list)]
               )

        name += 2

    # Create names on the x-axis
    plt.xticks([r + gap/2 for r in range(len(max(data, key=len)))], labels, rotation=45)    

    # Add title and axis names
    plt.title('%s per privacy level of synthetic data' % (QMname))
    plt.xlabel('Privacy level')
    plt.ylabel('%s score' % (QMname))

    # Create legend
    plt.legend(loc='lower right')

    plt.show()
    


def QMbar_two_DP_algs(QM, QMerror, QMname, QMori, labels, QMacc=None):
    
    """
    Create bar chart for the quality measure of synthetic datasets over the privacy levels when two different DP algorithms are included.
    QMacc is not required but can be passed to specify the synthetic accuracies to show at top of bars.
    QMacc should be passed as: [modelname, [syn accs], modelname, [syn_accs]].
    Input is similar to output of the quality measures, but the results of ms and pb should be appended in one list.
    QM is the result of ms + pb.
    QM error is the standard deviation of ms + pb.
    labels is de labels for the privacy levels, in the same order as the synthetic dataframes privacy levels are passed.
    """

    data = QM[1::2]
    errors = QMerror[1::2]
    color_list = ['firebrick', 'lightcoral', 'seagreen', 'mediumseagreen']
    gap = .8 / len(data)
    name=0
    
    fig = plt.figure(figsize=(10,7))
    
    bars = []


    for i, row in enumerate(data):
        X = np.arange(len(row))
        bar = plt.bar(X + i * gap, row,
        yerr = errors[i],
        align='center',
        capsize=5,              
        width = gap,  label=QM[name],
        color = color_list[i % len(color_list)]
               )
        
        bars.append(bar)
        name += 2

    # Create names on the x-axis
    plt.xticks([r + gap*1.4 for r in range(len(max(data, key=len)))], labels)  

    # Add title and axis names
    plt.title('%s Per Privacy Level of Synthetic Data' % (QMname))
    plt.xlabel('Privacy Level')
    plt.ylabel('%s score' % (QMname))
    
    # Add counts above the two bar graphs
    if QMacc != None:
        a=1
        b=1
        for bar in bars:
            for rect in bar:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height*1.01, '{}'.format(round(QMacc[a][b],2)), ha='center', va='bottom', color='black')
                b+=1
                if b > 5:
                    b=1
                    a+=2
        
    # Create legend   
    legend2 = plt.legend([bars[i] for i in range(0,len(bars))], ['LogReg {}'.format(round(QMori[1],2)), 'DecisionTree {}'.format(round(QMori[3],2)), 
                                                        'LogReg {}'.format(round(QMori[5],2)), 'DecisionTree {}'.format(round(QMori[7],2))],
                         loc='upper left', bbox_to_anchor=(1.05, 0.9))
        
    legend1 = plt.legend([bars[i] for i in range(0,len(bars), 2)], ["Marginal", "PrivBayes"], loc='upper left', bbox_to_anchor=(1.05, 1))
 
    plt.gca().add_artist(legend2)
    
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('green')

    plt.show()
    
    return fig


def SupportDistancebar(QM, QMerror, QMname, QMori, labels, QMacc=None):
    
    "Bar chart for the support distance of synthetic datasets over the privacy levels for two synthetic DP algorithms"
    
    
    data = QM[1::2]
    errors = QMerror[1::2]
    
    data=data[1::2]
    errors=errors[1::2]
    color_list = ['lightcoral', 'mediumseagreen', 'sandybrown' , 'orange', 'goldenrod']
    gap = 0.8 / len(data)
    name=2

    fig = plt.figure(figsize=(10,7))
    
    bars = []


    for i, row in enumerate(data):
        X = np.arange(len(row))
        bar = plt.bar(X + i * gap, row,
                      yerr = errors[i],
        align='center',
        capsize=5,
        width = gap,  label=QM[name],
        color = color_list[i % len(color_list)]
               )
        
        bars.append(bar)
        name += 2
        
    print(len(bars))

    # Create names on the x-axis
    plt.xticks([r + gap*0.5 for r in range(len(max(data, key=len)))], labels, rotation=45)    

    # Add title and axis names
    plt.title('%s Per Privacy Level of Synthetic Data' % (QMname))
    plt.xlabel('Privacy Level')
    plt.ylabel('%s score' % (QMname))
    
    # Add accuracies above the four bar graphs
    if QMacc != None:
        a=3
        b=1
        for bar in bars:
            for rect in bar:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height*1.01, '{}'.format(round(QMacc[a][b],2)), ha='center', va='bottom', color='black')
                b+=1
                if b > 5:
                    b=1
                    a+=4

    # Create legend   
    legend2 = plt.legend([bars[i] for i in range(0,len(bars))], ['Marginal DecisionTree {}'.format(round(QMori[3],2)), 
                                                        'PrivBayes DecisionTree {}'.format(round(QMori[7],2))], loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.gca().add_artist(legend2)

    plt.show()
    
    return fig
