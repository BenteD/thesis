#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from dython import nominal
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from synthesis.evaluation import visual, metrics, utility_evaluator
from numpy import linalg as LA
import math



## Feature distributions and distances between feature distributions



# Get average jensen-shannon distance over whole dataset
def calculateJSdistance(df_ori, privacy_levels):
    avgJSdistances=[]
    stdJSdistances=[]
    for df_syns in privacy_levels:
        JSdistances = []
        for df_syn in df_syns:
            features, feature_distances = metrics.feature_distances(df_ori, df_syn)
            JS = sum(feature_distances)/len(features)
            JSdistances.append(JS)
        avgJS = np.mean(JSdistances)
        stdJS = np.std(JSdistances)
        avgJSdistances.append(avgJS)
        stdJSdistances.append(stdJS)
    
    return avgJSdistances, stdJSdistances



def plotJSdistance(df_ori, privacy_levels, labels):
    
    avg, std = calculateJSdistance(df_ori, privacy_levels)
    
    # Line graph
    x = labels
    y = avg
    yerr = std
    
    plt.errorbar(x, y, yerr, capsize=3, marker='o')

    plt.xlabel('Privacy Level of Synthetic Dataset')
    plt.ylabel('Average Jensen-Shannon Distance')

    plt.title('Average Jensen-Shannon Distance over Privacy Levels')

    plt.show()



def feature_distributions(df_ori, df_syns, df_names, cont_vars=None, colors=None):
    
    """
    Creates feature disribution plots for the features of the given dataframes.
    For continious features the plots are kernel density estimations (probability denisty estimation).
    For other features the plots are normalized probability plots.
    Enter list continious variables if they are present in the datasets.
    Names of the dataframes should be given in a list for the df_names argument with original dataframe first 
    and synthetic dataframes next in the same order as df_syns for a correct overview.
    df_syns is a list of dataframes. Can be different privacy levels or one privacy level with multiple synthetic datasets.
    Own colors can be given as an argument, but not required. Required if df_syns > 6. E.g. sns.color_palette("GnBu").
    """
    
    if cont_vars == None:
        len_cont_vars = 0
        not_cont_vars = df_ori.columns.tolist()
    else:
        len_cont_vars = len(cont_vars)
        not_cont_vars = list(set(df_ori.columns.tolist()) - set(cont_vars))
    
    print("continuous variables are:", cont_vars)
    print("other variables are:", not_cont_vars)
    
    # Create one list for all the dataframes, with original data as first dataframe
    df_ori_syns = [df_ori] + df_syns
    
    # Define colors
    if colors == None:
        colors=["darkorange", "#115e67", "teal", "mediumseagreen",  "#86bf91", "khaki", "skyblue"]

    # Create number of rows for subplots as many as numerical variables there are (or categorical, depending which one are more present)
    if len_cont_vars >= (len(df_ori_syns[-1].columns.tolist()) - len_cont_vars):
        rows = len_cont_vars
    else:
        rows = len(df_ori_syns[0].columns) - len_cont_vars

    # Create subplots grid    
    f, axes = plt.subplots(rows, 2, figsize=(15, (rows*5)))

    # Set titles for numerical and categorical features (left and right side)
    f.suptitle("Distributions of the features (note: the value -1 represents missing values for continuous)", y=0.92)
    axes[0, 0].set_title('Probability density plots for continious numerical features')
    axes[0, 1].set_title('Normalized count plots for categorical features')

    # Create probability density plots for numerical features
    a=0
    if cont_vars != None:
        for column in cont_vars:
            i=1
            for df in df_syns:
                df=df.replace(np.nan, -1) # NaN values replaced by -1 to include them in the density histogram
                ax=sns.distplot(df[column] , color=colors[i], ax=axes[a, 0], label=df_names[i], hist=False)
                ax.get_legend().set_visible(False)
                i+=1
            # Original dataset last so it is visualized on top    
            df=df_ori.replace(np.nan, -1) # NaN values replaced by -1 to include them in the density histogram
            ax=sns.distplot(df[column] , color=colors[0], ax=axes[a, 0], label=df_names[0], hist=False)
            ax.get_legend().set_visible(False)
            a+=1

    # Create probability bar plots for categorical features
    b=0
    for column in not_cont_vars:
        z=0
        df_mixed = pd.DataFrame() # Create new dataframe for one barplot for all dataframes per feature
        for df in df_ori_syns:
            df=df.replace(np.nan, 'NaN') # NaN values replaced by string to include them in the probability bar chart (density not possible for non numerical features)
            df_mixed[z]=df[column]
            z+=1
        df_mixed.columns = df_names
        df_mixed = df_mixed.stack().reset_index() # Stack such that dataframe can be used in seaborn barplot
        df_mixed.columns = ['x', 'dataset', 'y']
        ax=sns.barplot(x='y', y='y', data=df_mixed, hue='dataset', palette=colors, ax=axes[b, 1], estimator=lambda x: len(x) / len(df), orient="v")
        ax.yaxis.label.set_visible(False)
        ax.get_legend().set_visible(False)
        ax.set_xlabel(column)
        b+=1
    
    # Add one legend for all subplots    
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5,0.875), bbox_transform=f.transFigure)

    # Remove subplots that are not used
    for x in range(a, rows):
        axes[x,0].set_axis_off()
    for x in range(b, rows):
        axes[x,1].set_axis_off()



## Correlations and correlation distances



def correlation_matrices(df_ori, df_syns, df_names, nominal_cols='auto', plot=True):
    
    """
    Computes the correlation matrices for the datasets.
    If argument plot is set to True, it plots correlation matrix heatmaps for the inserted dataframes.
    Names of the dataframes should be given in a list for the df_names argument with original data first 
    and synthetic dataframes next in the same order as df_syns for a clear overview.
    df_syns is a list of dataframes. Can be different privacy levels or one privacy level with multiple synthetic datasets.
    Default value for nominal_cols is 'auto'. This identifies nominal columns automatically. None means no nominal columns.
    If nominal columns are present they can be specified here, but it is not a required argument.
    """
    
    # Create one list for all the dataframes, with original data as first dataframe
    df_ori_syns = [df_ori] + df_syns
    
    # Plot the correlation matrices if plot is set True
    if plot == True:
        i=0
        for df in df_ori_syns:
            print(df_names[i])
            nominal.associations(df.copy(),
                                 nominal_columns = nominal_cols, theil_u = True, nan_strategy='drop_samples', mark_columns=True, 
                                 figsize = (7,7)
                                )
            i+=1



# For one version per privacy level
def corr_fro_norm(df_ori, df_syns, nominal_cols='auto'): 
    
    """
    Creates correlation matrices for inserted dataframes and compares them with the original dataframe.
    Returns the pair-wise euclidean distance between the two correlation matrices.
    Default value for nominal_cols is 'auto'. This identifies nominal columns automatically. None means no nominal columns.
    If nominal columns are present they can be specified here, but it is not a required argument.
    """
    
    ori = nominal.compute_associations(df_ori.copy(), 
                             nominal_columns=nominal_cols, mark_columns=True, theil_u=True, nan_strategy='drop_samples',
                            ) 
    
    fro_ori_syn = []
    for df in df_syns:
        df_syn = nominal.compute_associations(df.copy(), 
                             nominal_columns=nominal_cols, mark_columns=True, theil_u=True, nan_strategy='drop_samples',
                            )
        fro_norm = LA.norm(ori-df_syn, 'fro')
        fro_ori_syn.append(fro_norm)
    
    return fro_ori_syn



# For multiple versions per privacy level
def avg_corr_fro_norm(df_ori, privacy_levels, nominal_cols):
    
    """
    Does the same as corr_fro_norm but for multiple synthetic datasets per privacy level.
    """
    
    avgCorrnorm = []
    stdCorrnorm = []
    for df_syns in privacy_levels:
        fro_ori_syns = corr_fro_norm(df_ori, df_syns, nominal_cols)
        avgcorr = np.mean(fro_ori_syns)
        stdcorr = np.std(fro_ori_syns)
        avgCorrnorm.append(avgcorr)
        stdCorrnorm.append(stdcorr)
    
    return avgCorrnorm, stdCorrnorm



def plotCorrdistance(df_ori, privacy_levels, labels, nominal_cols):
    
    """
    Plots the results of avg_corr_fro_norm.
    """
    
    avg, std = avg_corr_fro_norm(df_ori, privacy_levels, nominal_cols)
    
    # Line graph
    x = labels
    y = avg
    yerr = std
    
    plt.errorbar(x, y, yerr, capsize=3, marker='o')

    plt.xlabel('Privacy Level of Synthetic Dataset')
    plt.ylabel('Frobenius Norm of Correlation Matrices')

    plt.title('Average Frobenius Norm over Privacy Levels')
    
    plt.show()


## Combined results plots

def plotJSdistances_two_algs(ms_avgJSdistances, ms_stdJSdistances, pb_avgJSdistances, pb_stdJSdistances, labels):

    # JS distances
    # Line graph with both algorithms
    
    fig = plt.figure()
    
    x = labels
    y1 = ms_avgJSdistances
    yerr1 = ms_stdJSdistances
    y2 = pb_avgJSdistances
    yerr2 = pb_stdJSdistances

    m = plt.errorbar(x, y1, yerr1, capsize=3, marker='o', label='Marginal', color="red")
    pb = plt.errorbar(x, y2, yerr2, capsize=3, marker='o', label='PrivBayes', color="green")

    plt.xlabel('Privacy Level of Synthetic Dataset')
    plt.ylabel('Average Jensen-Shannon Distance')

    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.title('Average Jensen-Shannon Distance over Privacy Levels')

    plt.show()
    
    return fig
    
    
def plotCorrdistances_two_algs(ms_avg_corr, ms_std_corr, pb_avg_corr, pb_std_corr, labels):

    # correlation euclidean distances/frobenius norm
    # Line graph with both algorithms
    
    fig = plt.figure()
    
    x = labels
    y1 = ms_avg_corr
    yerr1 = ms_std_corr
    y2 = pb_avg_corr
    yerr2 = pb_std_corr

    m = plt.errorbar(x, y1, yerr1, capsize=3, marker='o', label='Marginal', color="red")
    pb = plt.errorbar(x, y2, yerr2, capsize=3, marker='o', label='PrivBayes', color="green")

    plt.xlabel('Privacy Level of Synthetic Dataset')
    plt.ylabel('Frobenius Norm of Correlation Matrices')

    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.title('Average Frobenius Norm of the Difference Between Synthetic and Original Correlations over Privacy Levels')

    plt.show()
    
    return fig