import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from env import user, host, password

def wrangle_telco():

    def get_db_url(db):
        return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    query='''
    SELECT customer_id, monthly_charges, tenure, total_charges 
    FROM customers 
    WHERE contract_type_id = 3;
    '''
    df = pd.read_sql(query, get_db_url('telco_churn'))

    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df['total_charges'] = df['total_charges'].astype(float)
    df = df.dropna()
    return df
