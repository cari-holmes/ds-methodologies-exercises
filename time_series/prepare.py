import pandas as pd
import acquire


def prepare_sale():
    df = acquire.get_all_data(use_cache=True)
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values(by='sale_date').set_index('sale_date')
    df['month'] = df.index.strftime('%m-%b')
    df['day_of_week'] = df.index.strftime('%w-%A')
    df['total_sales'] = df['sale_amount'] * df['item_price']
    sales_sum = df.resample("D")[['total_sales']].sum()
    sales_sum['sales_differences'] = sales_sum['total_sales'].diff()
    return df


# def sale_distributions(df):
#     df = acquire.get_all_data(use_cache=True)
#     df = df[['sale_amount']].plot()
#     df = df[['item_price']].plot()
#     df = df[['sale_date']].plot()
#     return df


def prepare_ops():
    ops = acquire.get_opsd_data(use_cache=True)
    ops['Date'] = pd.to_datetime(ops['Date'])
    ops = ops.set_index('Date')
    ops['month'] = ops.index.month
    ops['year'] = ops.index.year
    return ops


# def ops_distributions(df:
#     ops = acquire.get_opsd_data(use_cache=True)
#     ops = ops[['Consumption']].plot()
#     ops = ops[['Wind']].plot()
#     ops = ops[['Solar']].plot()
#     ops = ops[['Wind+Solar']].plot()
#     return ops
