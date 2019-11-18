# 1. Using the code from the lesson as a guide, create a dataframe named items that has all of the data for items.

# 2. Do the same thing, but for stores.

# 3. Extract the data for sales. There are a lot of pages of data here, so your code will need to be a little more complex. Your code should continue fetching data from the next page until all of the data is extracted.

# 4. Save the data in your files to local csv files so that it will be faster to access in the future.

# 5. Combine the data from your three separate dataframes into one large dataframe.

# 6. Acquire the Open Power Systems Data for Germany, which has been rapidly expanding its renewable energy production in recent years. The data set includes country-wide totals of electricity consumption, wind power production, and solar power production for 2006-2017. You can get the data here: https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv

# 7. Make sure all the work that you have done above is reproducible. That is, you should put the code above into separate functions in the acquire.py file and be able to re-run the functions and get the same data.

from os import path

import requests
import pandas as pd

BASE_URL = 'https://python.zach.lol'
API_BASE = BASE_URL + '/api/v1'

def get_store_data_from_api():
    url = API_BASE + '/stores'
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['payload']['stores'])

#get_store_data_from_api()

def get_item_data_from_api():
    url = API_BASE + '/items'
    response = requests.get(url)
    data = response.json()

    stores = data['payload']['items']

    while data['payload']['next_page'] is not None:
        print('Fetching page {} of {}'.format(data['payload']['page'] + 1, data['payload']['max_page']))
        url = BASE_URL + data['payload']['next_page']
        response = requests.get(url)
        data = response.json()
        stores += data['payload']['items']

    return pd.DataFrame(stores)

#get_item_data_from_api()

def get_sale_data_from_api():
    url = API_BASE + '/sales'
    response = requests.get(url)
    data = response.json()

    stores = data['payload']['sales']

    while data['payload']['next_page'] is not None:
        print('Fetching page {} of {}'.format(data['payload']['page'] + 1, data['payload']['max_page']))
        url = BASE_URL + data['payload']['next_page']
        response = requests.get(url)
        data = response.json()
        stores += data['payload']['sales']

    return pd.DataFrame(stores)

#get_sale_data_from_api()

def get_store_data(use_cache=True):
    if use_cache and path.exists('stores.csv'):
        return pd.read_csv('stores.csv')
    df = get_store_data_from_api()
    df.to_csv('stores.csv', index=False)
    return df

#get_store_data(use_cache=True)

def get_item_data(use_cache=True):
    if use_cache and path.exists('items.csv'):
        return pd.read_csv('items.csv')
    df = get_item_data_from_api()
    df.to_csv('items.csv', index=False)
    return df

#get_item_data(use_cache=True)

def get_sale_data(use_cache=True):
    if use_cache and path.exists('sales.csv'):
        return pd.read_csv('sales.csv')
    df = get_sale_data_from_api()
    df.to_csv('sales.csv', index=False)
    return df

#get_sale_data(use_cache=True)

def get_opsd_data(use_cache=True):
    if use_cache and path.exists('opsd.csv'):
        return pd.read_csv('opsd.csv')
    df = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
    df.to_csv('opsd.csv', index=False)
    return df

#get_opsd_data(use_cache=True)


def get_all_data(use_cache=True):
    sales = get_sale_data()
    items = get_item_data()
    stores = get_store_data()

    sales = sales.rename(columns={'item': 'item_id', 'store': 'store_id'})

    return sales.merge(items, on='item_id').merge(stores, on='store_id')

#df = get_all_data(use_cache=True)

#df.head()
