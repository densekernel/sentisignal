import pandas as pd
import numpy as np 
import pylab as P 
import ast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import scipy.stats as s
import statsmodels.api as sm

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from datetime import datetime
from yahoo_finance import Share
from pandas_datareader import data, wb
from statsmodels.graphics.api import qqplot

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

#preprocess sentiment data with additional columns for:
# log_bullishness, log_bull_bear_ratio, TISf?, RTISf?, 
# difference in total scanned message, difference in (sum(bullish + bearish msgs))
def preprocess_data_sentiment(df):
    # log bull messages
    df['LOG_BULL_RETURN'] = np.log(1+df['BULL_SCORED_MESSAGES']).diff()
    # log bear messages
    df['LOG_BEAR_RETURN'] = np.log(1+df['BEAR_SCORED_MESSAGES']).diff()
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

#preprocess finance data with additional columns for:
# log_return, volatility, log_volume_diff
def preprocess_data_finance(df):
    # log return
    df['LOG_RETURN'] = np.log(1 + df['ADJ CLOSE'].pct_change())
    # volatitility
    df['VOLATILITY'] = df['HIGH'] - df['LOW']
    # difference in volume
    df['LOG_VOLUME_DIFF'] = np.log(df['VOLUME'].diff())
    # return df

# merge sentiment and finance data
# usually called with F, F, T
def merge_sentiment_finance(data_sentiment, data_finance, with_symbol, fill_finance, fill_sentiment):
    if with_symbol:
#         return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='left')
        if fill_finance:
            return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='left')
        if fill_sentiment:
            return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='right')
    else:
        if fill_finance:
            return pd.merge(data_sentiment, data_finance, on=['DATE'], how='left')
        if fill_sentiment:
            return pd.merge(data_sentiment, data_finance, on=['DATE'], how='right')

# descriptive statistics
# provide a range of graphics for descriptive statistics
def check_pdf(df):
    df_num = df.select_dtypes(include=[np.float, np.int])
    for index in df_num.columns:
        try:
            if index in ['LOG_BULL_RETURN', 'LOG_BEAR_RETURN','RTISf', 'TOTAL_SCANNED_MESSAGES_DIFF', 'TOTAL_SENTIMENT_MESSAGES_DIFF']:

                h = df_num[index][1:].sort_values().values
                fit = s.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
                
                P.plot(h,fit,'-o')
                P.hist(h,normed=True)      #use this to draw histogram of your data
                P.title(index)
                P.show()        

                # fig = sm.graphics.tsa.plot_acf(df_num[index][1:],lags=40)
                # plt.title(index)
            elif index in ['LOG_BULL_BEAR_RATIO']:
        
                h = df_num[index][1:].sort_values().values
                fit = s.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
                
                P.plot(h,fit,'-o')
                P.hist(h,normed=True)      #use this to draw histogram of your data
                P.title(index)
                P.show() 

            else: 
                
                h = df_num[index][1:].sort_values().values
                fit = s.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
                
                P.plot(h,fit,'-o')
                P.hist(h,normed=True)      #use this to draw histogram of your data
                P.title(index)
                P.show()
        except:
            print "error" 

# check autocorrelation
# provide a range of graphics which diagramatically show spikes for autocorrelation 
def check_acf(df):
    df_num = df.select_dtypes(include=[np.float, np.int])
    for index in df_num.columns:
        plt.figure(figsize=(8,10))
        if index in ['LOG_BULL_RETURN', 'LOG_BEAR_RETURN','RTISf', 'TOTAL_SCANNED_MESSAGES_DIFF', 'TOTAL_SENTIMENT_MESSAGES_DIFF']:
            fig = sm.graphics.tsa.plot_acf(df_num[index][1:],lags=40)
            plt.title(index)
        elif index in ['LOG_BULL_BEAR_RATIO']:
            fig = sm.graphics.tsa.plot_acf(df_num[index][2:],lags=40)
            plt.title(index)
        else: 
            fig = sm.graphics.tsa.plot_acf(df_num[index],lags=40)
            plt.title(index)
    return fig

# apply rollling window to data
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
       'BEAR_SCORED_MESSAGES','LOG_BULLISHNESS','TISf','RTISf'] ,columns = ['LOG_RETURN','VOLUME','VOLATILITY'])
    for column in resCorr.index:
        resCorr.ix[i,0] = df['LOG_RETURN'].corr(df[column])
        resCorr.ix[i,1] = df['VOLUME'].corr(df[column])
        resCorr.ix[i,2] = df['VOLATILITY'].corr(df[column])
        i += 1
    return resCorr

# Sturges formula: k = log_2 n + 1  
def sturges_bin(df):
    n = df.count()[1]
    return math.ceil(np.log2(n)+1)
# Rice Rule k = 2 n^{1/3}
def rice_bin(df):
    n = df.count()[1]
    return math.ceil(2*n**(1/3)) 
# Doanes formula for non-normal data.
def doane_bin(data):
    n = data.count()
    std = np.std(data)
    g = abs(s.moment(n,3)/(std**3))
    u = math.sqrt(6*(n-1)/((n+1)*(n+3)))
    return round(1 + np.log2(n) + np.log2(1+g/u))

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
                   
