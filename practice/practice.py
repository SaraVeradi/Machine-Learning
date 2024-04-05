#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 17 21:12:27 2024
To be completed as the study of the book goes on
@author: Sara Veradi
Based on codes in Hands-On Book on Machine Learning
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%===========================================================================
# Loading data
# =============================================================================
# I can use other methods but I usually download files and the length 
# ,type and the names are different

data_adress = "~/Documents/ML/practice/"
bbc         = pd.read_csv( data_adress + "bbc.csv")
data        = bbc                      # data to makes the program more 
                                       # adaptable for other network data

# %%===========================================================================
#  Take a look at data   
# =============================================================================
# Suppose we also want to look at the lenth of posts

data["word_count"] = data["text"].str.split().str.len()
# This is actually faster than it seems
# data['totalwords'] = [len(x.split()) for x in news['text'].tolist()]

print(data.head())
print(data.info())
print(data.describe())


hist = data.hist(bins=18, figsize=(8,6))
plt.suptitle("bbc data")


#%% ===========================================================================
#  CREATING A TEST SET AND TRAIN SET
# =============================================================================
# randomly decide on test set and training set
def random_test_set(data,  test_set_ratio):
    '''set aside a ratio of data as test set and training set
    given the data and ratio as a number between 0 and 1
    The first output is test set, the second output is trainig set'''
    
    np.random.seed(47)
    shuffled_indx = np.random.permutation(len(data))
    test_size = int(test_set_ratio*len(data) )
    test_set_indx = shuffled_indx[: test_size]
    train_set_indx = shuffled_indx[test_size:]
    return data.iloc[test_set_indx], data.iloc[train_set_indx]

# uncomment this line if you want to use randomly chosen test set
# test_set, train_set = random_test_set(data, 0.2)

#%% =============================================================================
# or use scikit builtin functions to avoid seeing all data after repeting the 
# algoithm
# As you see beyound some levels there are barely any data
# If we wnat some stratification we should make sure there are enough data in
# each category
# =============================================================================

from sklearn.model_selection import train_test_split

test_set, train_set = train_test_split(data, test_size=0.5, random_state=47)

# HERE WE NEED MORE WORKS ON STRATIFICATION
# bbc["comments_cat"] = pd.cut(bbc["shares"], bins=[0., 100., 200., 300., np.inf], 
#                                             labels=[1,2,3,4])

# plt.figure()
# bbc["comments_cat"].value_counts().sort_index().plot.bar(rot=0, width=0.1, grid=True)
# plt.show()

#%% ===========================================================================
# VISUALIZING DATA
# CORRELATION : the data is not too big, so it is helpful to compute
# correlation coefficient
# =============================================================================

# make a copy of training set to keep the original data
news = train_set.copy()

# Studying correlation of data: here the numbe of features are 4

cor_matrix = news.corr()
print(cor_matrix)
print(cor_matrix["likes"].sort_values(ascending=False))

# Another way of studying correalations is by using pandas liblraries
# so we only get 16 charts if we simply plot every data_colums vs all others

from pandas.plotting import scatter_matrix

scatter_matrix(news, figsize=(12, 8))

# Shares vs likes, as it was with the cor_matrix, is the most promising
# So let's plot it

news.plot(kind="scatter", x="likes", y="shares", alpha=0.5,
           s=news["comments"]/100, label="size=comments", 
           c="word_count", cmap="jet", colorbar=True, legend=True)
plt.xlim((0,4000))
plt.ylim((0,250 ))
plt.show()

#%% =============================================================================
# Cleaning data
# =============================================================================
# We want to replace NUL values with the median value of data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# This kind of cleaning works with numerical values, so "text" is not useful
# so we create dataframes with only numerical values

news_num = news.select_dtypes(include=[np.number])
imputer.fit(news_num)

# See the median of each attribute stored in imputer.statistics_
# and double check it
print("====attributes' median=====")
print( imputer.statistics_)
print( news_num.median().values)

# Transform data by replacing the missing data with the related median
# The output is a Numpy array
# Let's wrap it into a datframe

X = imputer.transform(news_num)
news_tr = pd.DataFrame(X, columns=news_num.columns, index=news_num.index)

#%% =============================================================================
# Feature Scaling
# =============================================================================

# Since there are outliers in data, we may use Standardization
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
news_scale = std_scaler.fit_transform(news_num)














