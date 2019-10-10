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

df = wrangle_telco()
df.info()
df.describe()


print(train.shape)
print(test.shape)

from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
# split_my_data(X, y, train_pct)

X = df.drop(columns=['customer_id', 'total_charges'])
X
y = df.total_charges

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=123)
X_train, X_test, y_train, y_test

# standard_scaler()



# scale_inverse()

# uniform_scaler()

# gaussian_scaler()

# min_max_scaler()

# iqr_robust_scaler()
