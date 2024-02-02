#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:12:27 2024

@author: Hands-On Book on Machine Learning
"""
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
housing.hist(bins=50, figsize=(12, 8))
plt.show()


# =============================================================================
# creating a test set
# =============================================================================
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# =============================================================================
# 
# =============================================================================
import numpy as np
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True,
                                                           xlabel="Income category")
# plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.savefig("housing_income_cat_bar_plot")  # extra code
plt.show()

# =============================================================================
# sklearn.model_selection package
# that implement various strategies to split your dataset into a training set and a test set.
# Each splitter has a split() method that returns an iterator over different training/
# test splits of the same data. split() method yields the training and test indices, not the
# data its
# =============================================================================
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
    
strat_train_set, strat_test_set = strat_splits[0]

# OR
# trat_train_set, strat_test_set = train_test_split(
# housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

#You won’t use the income_cat column again, so you might as well drop it, reverting
# the data back to its origin

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
#%% =============================================================================
#l VISUALIZING THE DATA
# =============================================================================

# make a copy of the original data

housing = strat_train_set.copy()

# Because the dataset includes geographical 
# information (latitude and longitude), it is a
# good idea to create a scatterplot

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, c='green',
              alpha = 0.1, s=housing["population"] / 100)
plt.show()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
s=housing["population"] / 100, label="population",
c="median_house_value", cmap="jet", colorbar=True, alpha=0.6,
legend=True, sharex=False, figsize=(10, 7))
plt.show()


#%% =============================================================================
# # extra code – this cell generates the first figure in the chapter
# =============================================================================

# Download the California image
# filename = "california.png"
# if not (IMAGES_PATH / filename).is_file():
#     homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
#     url = homl3_root + "images/end_to_end_project/" + filename
#     print("Downloading", filename)
#     urllib.request.urlretrieve(url, IMAGES_PATH / filename)

# housing_renamed = housing.rename(columns={
#     "latitude": "Latitude", "longitude": "Longitude",
#     "population": "Population",
#     "median_house_value": "Median house value (ᴜsᴅ)"})
# housing_renamed.plot(
#              kind="scatter", x="Longitude", y="Latitude",
#              s=housing_renamed["Population"] / 100, label="Population",
#              c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
#              legend=True, sharex=False, figsize=(10, 7))

# california_img = plt.imread(IMAGES_PATH / filename)
# axis = -124.55, -113.95, 32.45, 42.05
# plt.axis(axis)
# plt.imshow(california_img, extent=axis)

# save_fig("california_housing_prices_plot")
# plt.show()


#%% =============================================================================
# Data correlations
# =============================================================================

# A two dimensional matrix that show s correlations of data categories

corr_matrix = housing.corr(numeric_only=True)

# correlations between "house valu" and other parameters

print (corr_matrix["median_house_value"].sort_values(ascending=False))

# Another way to check for correlation between attributes is to use the Pandas
# scatter_matrix() function, which plots every numerical attribute against every
# other numerical attribute. Since there are now 11 numerical attributes, you would
# get 112 = 121 plots, which would not fit on a page—so you decide to focus on a
# few promising attributes that seem most correlated with the m

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()


#%% ===========================================================================
# Experiment with Attribute Combinations
# =============================================================================

# What you really want is the number of rooms per household. Similarly,
# the total number of bedrooms by itself is not very useful: you probably want to
# compare it to the number of rooms. And the population per household also seems
# like an interesting attribute combination to look at. 

housing["rooms_per_house"]  = housing["total_rooms"]    / housing["households"]
housing["bedrooms_ratio"]   = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"]     / housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


#%% ===========================================================================
# Prepare the Data for Machine Learning Algorithms
# =============================================================================

# But first, revert to a clean training set (by copying strat_train_set once again). You
# should also separate the predictors and the labels, since you don’t necessarily want
# to apply the same transformations to the predictors and the target values (note that
# drop() creates a copy of the data and does not affect strat_train_set):

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


#%% ===========================================================================
# Clean the Data : you noticed earlier that the total_bedrooms
# attribute has some missing values.
# =============================================================================
# With respect to missing data
# 1. set rid of the corresponding districts.
# 2. Get rid of the whole attribute.
# 3. set the missing values to some value (zero, the mean, the median, etc.). This is3.
# called imputation.

housing.dropna(subset=["total_bedrooms"], inplace=True) # option 1 : function to Remove missing values.

housing.drop("total_bedrooms", axis=1) # option 2

median = housing["total_bedrooms"].median() # option 3

housing["total_bedrooms"].fillna(median, inplace=True)

# =============================================================================
# SimpleImputer class for cleaning data
# =============================================================================
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
# replacing missing data with the learnt median
X = imputer.transform(housing_num)