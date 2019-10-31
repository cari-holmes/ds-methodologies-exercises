import acquire
import pandas as pd
from util import get_db_url
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def prep_iris():
    df_iris = acquire.get_iris_data()
    df_iris = df_iris.drop(columns=['species_id'])
    df_iris = df_iris.drop(columns=['measurement_id'])
    df_iris = df_iris.rename(columns={'species_name':'species'})
    
    encoder = LabelEncoder()
    df_iris.species = encoder.fit_transform(df_iris.species)
    return df_iris


def prep_titanic():
    df = acquire.get_titanic_data()
    df.embark_town.fillna('Other', inplace=True)
    df.embarked.fillna('Unknown', inplace=True)
    df.drop(columns=['deck'], inplace=True)
    
    encoder = LabelEncoder()
    df.embarked = encoder.fit_transform(df.embarked)
    
    scaler = MinMaxScaler()
    df.age = scaler.fit_transform(df[['age']])
    
    scaler = MinMaxScaler()
    df.fare = scaler.fit_transform(df[['fare']])
    
    return df

### split my data before scaling, adjust these functions


