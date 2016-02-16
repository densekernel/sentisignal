import pandas as pd
import numpy as np 
import pylab as P 
import ast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from yahoo_finance import Share
from pandas_datareader import data, wb

plt.style.use('ggplot')

# function to subsample the large data sets for:
# specific date range: start_data - end_date ('YYYY-MM-DD')
# filters : 'SYMBOL' - list of symbols ['SYM1', SYM2']
#'SECTOR' - list of symbols ['Sec1', 'Sec2']
#'EXCHANGE' - list of symbols ['Exch1', 'Exch2']
def subsample_data(filename_data, filename_symbology, start_date, end_date, query_attribute, query_criteria):
    # read csv
    data = pd.read_csv(filename_data)
    # convert timestamps to datetime objects
    data['datetime'] = data['TIMESTAMP_UTC'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%SZ'))
    # query between start and end date
    data = data[(data['datetime'] > start_date) & (data['datetime'] < end_date)]
    # merge with symbology csv for additional info
    data_symbology = pd.read_csv(filename_symbology)
    # convert headers to uppercase for ease of use
    data_symbology.columns = [x.upper() for x in data_symbology.columns]
    data = pd.merge(data, data_symbology, left_on='SYMBOL', right_on='SYMBOL', how = "left")
    # perform filter query based on parameters
    data = data[data[query_attribute].isin(query_criteria)]
    # return dataframe
    return data
                   
