'''Our scenario continues:

As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
I need to do this within an average of $5.00 per customer.

Create split_scale.py that will contain the functions that follow. 
Each scaler function should create the object, fit and transform both train and test. 
They should return the scaler, train dataframe scaled, test dataframe scaled. 
Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe. 
Be sure to set a random state where applicable for reproducibility!
'''

import warnings 
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np
from wrangle import wrangle_telco
import env 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

df = wrangle_telco()
X = df.drop(columns=['customer_id', 'total_charges'])
y = pd.DataFrame(df['total_charges'])

### Splitting entire df ###

def split_my_data_whole(df, train_pct=.80, random_state=123):
    train, test = train_test_split(df, train_size=train_pct, random_state=random_state)
    return train, test

# Scaling Methods #
# Workflow:
#1. Create object and 
#2. Fit object
#3. Transform train and test

def standard_scalar(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaler, test_scaler

def scale_inverse(scaler, train_scaler, test_scaler):
    train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train.index.values])
    test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test.index.values])
    return scaler, train, test

def uniform_scaler(train, test):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def gaussian_scaler(train, test):
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def min_max_scaler():
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, test_scaled

### Splitting the df into 4 df's ###

'''
def split_my_data(X, y, train_pct=.80):
    X = df.drop(columns=['customer_id', 'total_charges'])
    y = pd.DataFrame(df['total_charges'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=123)
    return X_train, X_test, y_train, y_test

# Types of Scaling #

def standard_scaler(X_train, X_test, y_train, y_test):
    scalerX = StandardScaler().fit(X_train)
    scalery = StandardScaler().fit(y_train) 
    train_scaled_X = pd.DataFrame(scalerX.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    train_scaled_y = pd.DataFrame(scalery.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
    test_scaled_X = pd.DataFrame(scalerX.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])
    test_scaled_y = pd.DataFrame(scalery.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])
    return scalerX, scalery, train_scaled_X, train_scaled_y, test_scaled_X, test_scaled_y

def scale_inverse(scalerX, scalery, train_scaled_X, train_scaled_y, test_scaled_X, test_scaled_y):
    unscaledX = pd.DataFrame(scalerX.inverse_transform(train_scaled_X), columns=train_scaled_X.columns.values).set_index([X_train.index.values])
    unscaledy = pd.DataFrame(scalery.inverse_transform(train_scaled_y), columns=train_scaled_y.columns.values).set_index([y_train.index.values])
    test_X_unscaled = pd.DataFrame(scalerX.inverse_transform(test_scaled_X), columns=test_scaled_X.columns.values).set_index([X_test.index.values])
    test_y_unscaled = pd.DataFrame(scalery.inverse_transform(test_scaled_y), columns=test_scaled_y.columns.values).set_index([y_test.index.values])
    return unscaledX, unscaledy, test_X_unscaled, test_y_unscaled

def uniform_scaler(X_train, X_test, y_train, y_test):
    scalerX = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(X_train)
    scalery = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(y_train)
    train_uniform_X = pd.DataFrame(scalerX.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    train_uniform_y = pd.DataFrame(scalery.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
    test_uniform_X = pd.DataFrame(scalerX.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])
    test_uniform_y = pd.DataFrame(scalery.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])
    return scalerX, scalery, train_uniform_X, train_uniform_y, test_uniform_X, test_uniform_y

def gaussian_scaler(X_train, X_test, y_train, y_test):
    scalerX = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(X_train)
    scalery = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(y_train)
    train_yeo_X = pd.DataFrame(scalerX.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    train_yeo_y = pd.DataFrame(scalery.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
    test_yeo_X = pd.DataFrame(scalerX.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])
    test_yeo_y = pd.DataFrame(scalery.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])
    return scalerX, scalery, train_yeo_X, train_yeo_y, test_yeo_X, test_yeo_y

def min_max_scaler(X_train, X_test, y_train, y_test):
    scalerX = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)
    scalery = MinMaxScaler(copy=True, feature_range=(0,1)).fit(y_train)
    train_minmax_X = pd.DataFrame(scalerX.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    train_minmax_y = pd.DataFrame(scalery.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
    test_minmax_X = pd.DataFrame(scalerX.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])
    test_minmax_y = pd.DataFrame(scalery.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])
    return scalerX, scalery, train_minmax_X, train_minmax_y, test_minmax_X, test_minmax_y

def iqr_robust_scaler(X_train, X_test, y_train, y_test):
    scalerX = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(X_train)
    scalery = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(y_train)
    train_iqr_X = pd.DataFrame(scalerX.transform(X_train), columns=X_train.columns.values).set_index([X_train.index.values])
    train_iqr_y = pd.DataFrame(scalery.transform(y_train), columns=y_train.columns.values).set_index([y_train.index.values])
    test_iqr_X = pd.DataFrame(scalerX.transform(X_test), columns=X_test.columns.values).set_index([X_test.index.values])
    test_iqr_y = pd.DataFrame(scalery.transform(y_test), columns=y_test.columns.values).set_index([y_test.index.values])
    return scalerX, scalery, train_iqr_X, train_iqr_y, test_iqr_X, test_iqr_y
'''

