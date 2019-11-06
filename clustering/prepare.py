import pandas as pd
import scipy.stats as stats
import numpy as np

# Selects single unit properties out of data.

def zillow_single_unit(df):
    criteria_1 = df.propertylandusedesc == 'Single Family Residential'
    #criteria_2=df.unitcnt==1 | df.unitcnt.isna()
    #criteria_2=df.unitcnt==1 & calculatedfinishedsquarefeet>500
    criteria_2 = df.calculatedfinishedsquarefeet > 500
    df = df[(criteria_1) & (criteria_2)]
    return df

# Remove unwanted columns.

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)
    return df

# Remove rows and columns based on a minimum percentage for each row and column.

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# Combining both the previous functions together.

def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

# Fills missing values with 0's where it makes sense.

def fill_zero(df, cols):
    df.fillna(value=0, inplace=True)
    return df

# Removes outliers.

def remove_outliers_iqr(df, columns):
    for col in columns:
        q75, q25 = np.percentile(df[col], [75,25])
        ub = 3*stats.iqr(df[col]) + q75
        lb = q25 - 3*stats.iqr(df[col])
        df = df[df[col] <= ub]
        df = df[df[col] >= lb]
    return df

