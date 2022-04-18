# Importing libraries
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# read the data
class DataClean:

    df = pd.read_csv("../test_images_kaggle/images")
    df.shape
# shape of the pixel array
print(df.shape)
print(df.dtypes)

# Clean each image in the folder
# Remove duplicates
# Handle missing data as a %
# select numeric and non numeric columns
def split_data():
    
    df_non_numeric = df.select_dtypes(exclude=[np.number])
    non_numeric_cols = df_non_numeric.columns.values

    df_numeric = df.select_dtypes(include=[np.number])
    numeric_cols = df_numeric.columns.values



#values missing in each column
def missing_data():
    col_list = list()
    value_list = list()

    for col in df.columns:

        pc_missing = np.mean(df[col].isnull())*100
        col_list.append(col)
        value_list.append(pc_missing)
        pc_missing_df = pd.DataFrame()
        pc_missing_df['col'] = col_list
        pc_missing_df['pc_missing'] = value_list

# input missing values?

# append rows to deal with duplicates 

#  Outliers 