import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import wrangle

#############################################################

#        EXPLORE.PY        #

def plot_variable_pairs(train, cont_var, cat_var):
# columns    
    cat_var = ['bedrooms', 'bathrooms', 'fips']
    cont_var = ['area', 'taxamount', 'tax_value', 'year_built']

#plots  
    sns.lmplot(x='tax_value', y='area', data=train.sample(1000), scatter=True)
    sns.lmplot(x='tax_value', y='year_built', data=train.sample(1000), scatter=True)
    sns.lmplot(x='tax_value', y='taxamount', data=train.sample(1000), scatter=True)


     
    return train, cont_var, cat_var


def plot_categorical_and_continuous_vars(train, cont_var, cat_var):
    
    # columns    
    cat_var = ['bedrooms', 'bathrooms', 'fips']
    cont_var = ['square_feet', 'tax_amount', 'tax_value', 'year_built']
    
    # plots
    sns.boxplot(x='bedrooms', y='tax_value', data=train.sample(1000))
    plt.show()
    sns.violinplot(x='bathrooms', y='tax_value', data=train.sample(1000))
    plt.show()
    sns.barplot(x='fips', y='tax_value', data=train.sample(1000))
    plt.show()
    
    return train, cont_var, cat_var






