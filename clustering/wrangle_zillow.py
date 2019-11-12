import acquire
import prepare

# Wrangle prepared zillow data

def wrangle_zillow_data():
    df = acquire.acquire_zillow()
    df = prepare.zillow_single_unit(df)
    df = prepare.remove_columns(df,['calculatedbathnbr','finishedsquarefeet12',\
        'fullbathcnt','propertycountylandusecode','unitcnt','structuretaxvaluedollarcnt',\
        'landtaxvaluedollarcnt','assessmentyear','propertyzoningdesc'])
    df = prepare.handle_missing_values(df)
    df.dropna(inplace = True)
    return df

