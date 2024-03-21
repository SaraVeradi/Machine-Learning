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

# %%===========================================================================
# Loading data
# =============================================================================
# I can use other methods but I usually download files and the length 
# ,type and the names are different

data_adress = "~/Documents/ML/practice/"
aljaz  = pd.read_csv( data_adress + "al_jazeera.csv")
reuter = pd.read_csv( data_adress + "reuters.csv")
bbc    = pd.read_csv( data_adress + "bbc.csv")
cnn    = pd.read_csv( data_adress + "cnn.csv")


# df = pd.DataFrame( [cnn["likes"], cnn["comments"],cnn["shares"]], columns=cnn.columns[1:])
# %%===========================================================================
#  Take a look at data   
# =============================================================================

print(bbc.head())
print(bbc.info())
print(bbc.describe())

# fig, axs = plt.subplots(3, 3, figsize=(12,8))
# names = cnn.columns
# axs[0,1].hist(cnn["likes"], bins=10)

hist=bbc.hist(bins=10, figsize=(8,6))
plt.suptitle("bbc data")
























