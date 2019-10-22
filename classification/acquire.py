import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from util import get_db_url


# In a new python module, acquire.py:

# get_titanic_data: returns the titanic data from the codeup data science database as a pandas data frame.
def get_titanic_data():
    query='''
    SELECT * FROM passengers;
    '''
    df = pd.read_sql(query, get_db_url('titanic_db'))
    return df

df_titanic = get_titanic_data()
df_titanic.head()

# get_iris_data: returns the data from the iris_db on the codeup data science database as a pandas data frame. The returned data frame should include the actual name of the species in addition to the species_ids.
def get_iris_data():
    query='''
    SELECT * FROM measurements
    JOIN species USING (species_id);
    '''
    df = pd.read_sql(query, get_db_url('iris_db'))
    return df

df_iris = get_iris_data()
df_iris.head()
