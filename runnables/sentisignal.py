import pandas as pd
import numpy as np 
import pylab as P 
import ast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn import metrics
from datetime import datetime
from yahoo_finance import Share
from pandas_datareader import data, wb

plt.style.use('ggplot')

# function to subsample the large data sets for:
# specific date range: start_data - end_date ('YYYY-MM-DD')
# filters : 'SYMBOL' - list of symbols ['SYM1', SYM2']
#'SECTOR' - list of symbols ['Sec1', 'Sec2']
#'EXCHANGE' - list of symbols ['Exch1', 'Exch2']
def subsample_data(filename_data, filename_symbology, dir_pickle, start_date, end_date, query_attribute, query_criteria, include_avg):
    query_criteria_filename = '-'.join(query_criteria[:3])
    pickle_name = dir_pickle+'pickle_sentiment_'+start_date+'_'+end_date+'_'+query_attribute+'_'+query_criteria_filename+'.p'
    try: 
        data = pd.read_pickle(pickle_name)
        print("Loaded from pre-created pickle")
    except:
        print("Subsampling data from csv")
        # try to read first from pickle
        # read csv
        data = pd.read_csv(filename_data)
        # merge with symbology csv for additional info
        data_symbology = pd.read_csv(filename_symbology)
        # convert headers to uppercase for ease of use
        data_symbology.columns = [x.upper() for x in data_symbology.columns]
        data = pd.merge(data, data_symbology, left_on='SYMBOL', right_on='SYMBOL', how = "left")
        # perform filter query based on parameters
        data = data[data[query_attribute].isin(query_criteria)]
        # convert timestamps to datetime objects
        data['DATE'] = data['TIMESTAMP_UTC'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%SZ'))
        data['DATE'] = data['DATE'].apply(lambda x: x.strftime('%x'))
        data['DATE'] = data['DATE'].apply(lambda x: pd.to_datetime(x))
        # query between start and end date
        data = data[(data['DATE'] > start_date) & (data['DATE'] < end_date)]
        # remove avg
        if include_avg == False:
            avg_cols = [col for col in data.columns if 'AVG' in col]
            data.drop(avg_cols,inplace=True,axis=1)
        # save as pickle
        data.to_pickle(pickle_name)
    # return dataframe
    return data

#preprocess sentiment data with additional columns for:
# log_bullishness, log_bull_bear_ratio, TISf?, RTISf?, 
# difference in total scanned message, difference in (sum(bullish + bearish msgs))
def preprocess_data_sentiment(df):
    # log bullishness
    df['LOG_BULLISHNESS'] = np.log(1 + df['BULL_SCORED_MESSAGES']) - np.log(1 + df['BEAR_SCORED_MESSAGES'])
    # log bull bear ratio
    df['LOG_BULL_BEAR_RATIO'] = np.log(df['BULL_SCORED_MESSAGES']) / np.log(df['BEAR_SCORED_MESSAGES'])
    # TISf
    df['TISf'] = (1+df['BULL_SCORED_MESSAGES'])/(1+ df['BULL_SCORED_MESSAGES']+df['BEAR_SCORED_MESSAGES'])
    # RTISf
    df['RTISf'] = ((1+df['BULL_SCORED_MESSAGES'])/(1+ df['BULL_SCORED_MESSAGES']+df['BEAR_SCORED_MESSAGES'])).pct_change()
    # diff in total scanned messages
    df['TOTAL_SCANNED_MESSAGES_DIFF'] = df['TOTAL_SCANNED_MESSAGES'].diff()
    # diff in total scanned messages
    df['TOTAL_SENTIMENT_MESSAGES_DIFF'] = (df['BULL_SCORED_MESSAGES']+df['BEAR_SCORED_MESSAGES']).diff()


    # return df

def preprocess_data_finance(df):
    # log return
    df['CHANGE'] = df['ADJ CLOSE'].pct_change()
    df['LOG_RETURN'] = np.log(1 + df['CHANGE'])
    # volatitility
    df['VOLATILITY'] = df['HIGH'] - df['LOW']
    # difference in volume
    df['VOLUME_DIFF'] = df['VOLUME'].diff()
    # return df

# function to return historical finance data
# takes a list of ticker symbols
# if sum_data_frame is true -> create a summed value to form an indic
# else create a concatenated dataframe
#
def get_data_finance(source, symbols, start_date, end_date, dir_pickle, sum_data_frame, sum_symbol):
    symbols_filename = '-'.join(symbols[:3])
    pickle_name = dir_pickle+'pickle_finance_'+start_date+'_'+end_date+'_'+symbols_filename+'.p'
    try: 
        data_finance = pd.read_pickle(pickle_name)
        print("Loaded from pre-created pickle")
    except:
        print("Scraping and saving data from Yahoo")
        # get finance data using pandas data reader
        # create df from first symbol
        try:
            data_finance = data.DataReader(symbols[0], source, start_date, end_date)
            # convert headers to uppercase for ease of use
            data_finance.columns = [x.upper() for x in data_finance.columns]
            # preprocess_data_finance(data_finance)
            if sum_data_frame:
                data_finance['SYMBOL'] = sum_symbol
            else:
                data_finance['SYMBOL'] = symbols[0]
        #         print(data_finance.head())
        except: 
            print("Unable to retrieve data for: " + symbols[0])
        # reset index (optional)
        # data_finance.reset_index(level=0, inplace=True)
        # loop through remaining symbols and either add or concatenate
        for symbol in symbols[1:]:
            print(symbol)
            try:
                symbol_finance = data.DataReader(symbol, source, start_date, end_date)
                # convert headers to uppercase for ease of use
                symbol_finance.columns = [x.upper() for x in symbol_finance.columns]
                # preprocess_data_finance(symbol_finance)
                # reset index (optional)
                # symbol_finance.reset_index(level=0, inplace=True)
                # sum dataframes
                if sum_data_frame:
                    symbol_finance['SYMBOL'] = sum_symbol
                    data_finance = data_finance + symbol_finance
                # vertically concat dataframes
                else:
                    symbol_finance['SYMBOL'] = symbol
                    data_finance = pd.concat([data_finance, symbol_finance], axis=0)
            except: 
                print("Unable to retrieve data for: " + symbol)
        # reset index (optional)
        data_finance.reset_index(level=0, inplace=True)
        # convert headers to uppercase for ease of use
        data_finance.columns = [x.upper() for x in data_finance.columns]
        # Set Date as index
        # data_finance.index = data_finance['DATE']
        # Sort by date
        data_finance = data_finance.sort_values(['DATE'], ascending=[True])
        data_finance.to_pickle(pickle_name)
    # return as dataframe
    return data_finance

def merge_sentiment_finance(data_sentiment, data_finance, with_symbol, fill_finance, fill_sentiment):
    if with_symbol:
#         return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='left')
        if fill_finance:
            return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='left')
        if fill_sentiment:
            return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='left')
    else:
        if fill_finance:
            return pd.merge(data_sentiment, data_finance, on=['DATE'], how='left')
        if fill_sentiment:
            return pd.merge(data_sentiment, data_finance, on=['DATE'], how='right')


def apply_rolling_window(df, width):
    df = df.ix[1:, :]
    df_num = df.select_dtypes(include=[np.float, np.int])
    df_non_num = df.select_dtypes(exclude=[np.float, np.int])
    df_num_window = pd.rolling_mean(df_num, width, min_periods=1)
    df_window = pd.concat([df_non_num, df_num_window], axis=1)
    return df_window

def correlation_analysis(df):
    i = 0
    resCorr = pd.DataFrame(index = ['BULLISH_INTENSITY','BEARISH_INTENSITY','BULL_MINUS_BEAR','BULL_SCORED_MESSAGES',
       'BEAR_SCORED_MESSAGES','LOG_BULLISHNESS','TISf','RTISf'] ,columns = ['LOG_RETURN','VOLUME'])
    for column in resCorr.index:
        resCorr.ix[i,0] = df['LOG_RETURN'].corr(df[column])
        resCorr.ix[i,1] = df['VOLUME'].corr(df[column])
        i += 1
    return resCorr

def calc_mutual_information(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi

def information_surplus(df, time_shift, varx, vary, bins):
    output = []
    for i in range(0, time_shift+1):
        shift_x = df[varx].shift(i)
        x = shift_x.ix[1+i:]
        y = df[vary].ix[1+i:]
        mi = calc_mutual_information(x, y, bins)
        if i == 0:
            mi_origin = mi
        output.append({'SHIFT': i, 'MUTUAL INFORMATION': mi, 'INFORMATION_SURPLUS_DIFF': mi - mi_origin, 'INFORMATION_SURPLUS_PCT': (mi - mi_origin) / mi_origin * 100})
    output_frame = pd.DataFrame(output)
    return output_frame
                   
