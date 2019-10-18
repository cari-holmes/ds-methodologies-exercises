# Load the tips dataset from either pydataset or seaborn.

from pydataset import data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from math import sqrt 
import warnings
warnings.filterwarnings('ignore')
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression

df = data('tips')
df.describe()

# Fit a linear regression model (ordinary least squares) and compute yhat, predictions of tip using total_bill. You may follow these steps to do that:
'''
- import the method from statsmodels: from statsmodels.formula.api import ols
- fit the model to your data, where x = total_bill and y = tip: regr = ols('y ~ x', data=df).fit()
- compute yhat, the predictions of tip using total_bill: df['yhat'] = regr.predict(df.x)
- Create a file evaluate.py that contains the following functions.
'''

x = df.total_bill
y = df.tip

regr = ols('tip ~ total_bill', data=df).fit()
df['yhat'] = regr.predict(pd.DataFrame(x))
regr.summary()
df.head()

# Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, and the dataframe as input and returns a residual plot. (hint: seaborn has an easy way to do this!)

def plot_residuals(x, y, df):
    return sns.residplot(x, y, df)

plot_residuals(x, y, df)

# Write a function, regression_errors(y, yhat), that takes in y and yhat, returns the sum of squared errors (SSE), explained sum of squares (ESS), total sum of squares (TSS), mean squared error (MSE) and root mean squared error (RMSE).

def regression_errors(y, yhat):
    MSE = mean_squared_error(y, df.yhat)
    SSE = MSE*len(df)
    RMSE = sqrt(MSE)
    ESS = sum((df.yhat - y.mean())**2)
    TSS = SSE + ESS
    return  SSE, ESS, TSS, MSE, RMSE  

regression_errors(y, df.yhat)

# Write a function, baseline_mean_errors(y), that takes in your target, y, computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and returns the error values (SSE, MSE, and RMSE).
## fix this baseline

def baseline_mean_errors(y):
    SSE_base = ((y.mean() - y)** 2).sum()
    MSE_base = SSE_base / len(y)
    RMSE_base = sqrt(MSE_base)
    return SSE_base, MSE_base, RMSE_base

baseline_mean_errors(y)


# Write a function, better_than_baseline(SSE), that returns true if your model performs better than the baseline, otherwise false.

SSE = ((df.yhat - y) ** 2).sum()
SSE_base = ((y.mean() - y)** 2).sum()

def better_than_baseline(SSE, SSE_base):
    SSE = ((df.yhat - y) ** 2).sum()
    SSE_base = ((y.mean() - y)** 2).sum()
    return SSE < SSE_base

better_than_baseline(SSE, SSE_base)

# Write a function, model_significance(ols_model), that takes the ols model as input and returns the amount of variance explained in your model, and the value telling you whether the correlation between the model and the tip value are statistically significant.

ols_model = ols('y ~ x', data=df).fit()

def model_significance(ols_model):
    r2 = ols_model.rsquared
    pval = ols_model.f_pvalue
    return r2, pval
    
model_significance(ols_model)
