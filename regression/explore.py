# Our scenario continues:
'''
    As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
    I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
    I need to do this within an average of $5.00 per customer.
'''

# Create a file, explore.py, that contains the following functions for exploring your variables (features & target).
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

import env
import wrangle
import split_scale 

df = wrangle.wrangle_telco()
df.head()

train, test = split_my_data_whole(df)
train.head(), test.head()
type(train)
type(test)

#1. Write a function, plot_variable_pairs(dataframe) that plots all of the pairwise relationships along with the regression line for each pair.
def plot_variable_pairs(df):

scaled_train, scaled_test = standard_scaler(train, test)
    
df_plt = sns.jointplot('monthly_charges', 'tenure', data=train, kind='reg')


j = sns.jointplot("exam1", "final_grade", data=train, kind='reg', height=5);
plt.show()
g = sns.PairGrid(train)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);

### jsut pick columns I want. 

#2. Write a function, months_to_years(tenure_months, df) that returns your dataframe with a new feature tenure_years, in complete years as a customer.
def months_to_years(tenure_months, df):


#3. Write a function, plot_categorical_and_continous_vars(categorical_var, continuous_var, df), that outputs 3 different plots for plotting a categorical variable with a continuous variable, e.g. tenure_years with total_charges. 
    # For ideas on effective ways to visualize categorical with continuous: https://datavizcatalogue.com/. You can then look into seaborn and matplotlib documentation for ways to create plots.
def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):



