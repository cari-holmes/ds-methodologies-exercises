# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: 
'''As a customer analyst, I want to know who has spent the most money with us over their lifetime. 
I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. 
I need to do this within an average of $5.00 per customer.'''

# The first step will be to acquire and prep the data. Do your work for this exercise in a file named wrangle.py.

import warnings 
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import user, host, password

def get_db_url(db):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

query='''
SELECT customer_id, monthly_charges, tenure, total_charges 
FROM customers 
WHERE contract_type_id = 3;
'''

df_telco = pd.read_sql(query, get_db_url('telco_churn'))
df_telco.head() 
df_telco.shape
df_telco.describe()
df_telco.info()
df_telco.total_charges.value_counts(sort=True)

#1. Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.

'''* change the dataframe from pandas *
# SELECT * FROM customers (SQL query)
# my_columns = df_telco[['customer_id', 'monthly_charges', 'tenure', 'total_charges', 'contract_type_id']]
# telco = my_columns[my_columns.contract_type_id == 3]
# telco.head()
'''

#2. Walk through the steps above using your new dataframe. You may handle the missing values however you feel is appropriate.

df_telco.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df_telco['total_charges'] = df_telco['total_charges'].astype(float)
print(df_telco.isnull().sum())
print(df_telco.columns[df_telco.isnull().any()])
df_telco = df_telco.dropna()
df_telco.dtypes 

#3. End with a python file wrangle.py that contains the function, wrangle_telco(), that will acquire the data and return a dataframe cleaned with no missing values.

def wrangle_telco():
    import warnings 
    warnings.filterwarnings('ignore')

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from env import user, host, password

    def get_db_url(db):
        return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    query='''
    SELECT customer_id, monthly_charges, tenure, total_charges 
    FROM customers 
    WHERE contract_type_id = 3;
    '''
    df_telco = pd.read_sql(query, get_db_url('telco_churn'))

    df_telco.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df_telco['total_charges'] = df_telco['total_charges'].astype(float)
    df_telco = df_telco.dropna()
    return df_telco

def wrangle_telco():
    