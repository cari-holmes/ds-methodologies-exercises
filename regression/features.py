### necessary imports, some scattered throughout the functions as well
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

### import env, wrangle function, split_scale function
import env
import wrangle
import split_scale

### 
df = wrangle.wrangle_telco()

df = df.drop(columns=['customer_id'])
df.head()
train, test = split_scale.split_my_data(df)

x_train = train.drop(columns=['total_charges'])
y_train = train[['total_charges']]
x_test = test.drop(columns=['total_charges'])
y_test = test[['total_charges']]

scaler, train_scaled, test_scaled = split_scale.standard_scaler(train, test)

x_train_scaled = train.drop(columns=['total_charges'])
y_train_scaled = train[['total_charges']]
x_test_scaled = test.drop(columns=['total_charges'])
y_test_scaled = test[['total_charges']]

# Our scenario continues:
'''
As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
I need to do this within an average of $5.00 per customer.
'''

#1. Write a function, select_kbest_fregression() that takes X_train, y_train and k as input (X_train and y_train should not be scaled!) and returns a list of the top k features.
from sklearn.feature_selection import SelectKBest, f_regression

def select_kbest_fregression(x_train, y_train, k):
    f_selector = SelectKBest(f_regression, k=k).fit(x_train, y_train)
    f_support = f_selector.get_support()
    f_feature = x_train.loc[:, f_support].columns.tolist()
    return f_feature

select_kbest_fregression(x_train, y_train, 2)

#2. Write a function, select_kbest_freg() that takes X_train, y_train (scaled) and k as input and returns a list of the top k features.

def select_kbest_freg(x_train, y_train, k):
    f_selector = SelectKBest(f_regression, k=k).fit(x_train, y_train)
    f_support = f_selector.get_support()
    f_feature = x_train.loc[:, f_support].columns.tolist()
    return f_feature

select_kbest_freg(x_train_scaled, y_train_scaled, 2)

#3. Write a function, ols_backward_elimination() that takes X_train and y_train (scaled) as input and returns selected features based on the ols backwards elimination method.
import statsmodels.api as sm 

def ols_backward_elimination(x_train, y_train):
    cols = list(x_train.columns)

    while (len(cols) > 0):
        x_1 = x_train[cols]
        model = sm.OLS(y_train,x_1).fit()
        p = pd.Series(model.pvalues)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break

    selected_features_BE = cols
    return selected_features_BE

ols_backward_elimination(x_train_scaled, y_train_scaled)


#4. Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns the coefficients for each feature, along with a plot of the features and their weights.
from sklearn.linear_model import LassoCV

def lasso_cv_coef(x_train, y_train):
    reg = LassoCV()
    reg.fit(x_train, y_train)
    coef = pd.Series(reg.coef_, index = x_train.columns)
    plot = sns.barplot(x = x_train.columns, y = reg.coef_)
    return coef, plot

lasso_cv_coef(x_train, y_train)


#5. Write 3 functions, the first computes the number of optimum features (n) using rfe, the second takes n as input and returns the top n features, and the third takes the list of the top n features as input and returns a new X_train and X_test dataframe with those top features , recursive_feature_elimination() that computes the optimum number of features (n) and returns the top n features.
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def optimal_n_features(x_train, y_train):
    features_range = range(1, len(x_train.columns)+1)
    high_score = 0
    number_of_features = 0
    score_list = []
    for n in features_range:
        model = LinearRegression()
        train_rfe = RFE(model, n).fit_transform(x_train, y_train)
        model.fit(train_rfe, y_train)
        score = model.score(train_rfe, y_train)
        score_list.append(score)
        if(score > high_score):
            high_score = score
            number_of_features = n

    return number_of_features, score

optimal_n_features(x_train, y_train)



def top_features(x_train, y_train, number_of_features):
    cols = list(x_train.columns)
    model = LinearRegression()
    rfe = RFE(model, number_of_features)
    x_rfe = rfe.fit_transform(x_train,y_train)  
    model.fit(x_rfe,y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    
    return selected_features_rfe

top_features(x_train, y_train, 2)

#def list_top_features()