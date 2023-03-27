import pandas as pd
import numpy as np
import env
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# For Zillow Data

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_zillow():
    '''
    Grab our data from path and read as csv
    '''
    
    df = pd.read_sql('SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017 join predictions_2017 using(parcelid) where propertylandusetypeid = 261;', get_connection('zillow'))

    return(df)
    
def clean_zillow(df):
    '''
    Takes in a df of zillow_data and cleans the data 
    appropriatly by dropping nulls.

    return: df, a cleaned pandas data frame.
    '''
    
    # Instead of using dummies to seperate contracts use, 
    # df[['Contract']].value_counts()
    # Use a SQL querry
    
    df = df
    df = df.dropna()
    return df
    
