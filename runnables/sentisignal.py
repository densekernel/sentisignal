import pandas as pd
import numpy as np 
import pylab as P 
import ast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import scipy.stats as s
import statsmodels.api as sm
import pprint
import seaborn as sns

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from datetime import datetime
from yahoo_finance import Share
from pandas_datareader import data, wb
from statsmodels.graphics.api import qqplot
from operator import itemgetter
from decimal import *
from sklearn.neighbors.kde import KernelDensity

# date plot config

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

# plt.style.use('ggplot')
sns.set_style("white")
# sns.set_context("notebook")

# function to subsample the large data sets for:
# specific date range: start_data - end_date ('YYYY-MM-DD')
# filters : 'SYMBOL' - list of symbols ['SYM1', SYM2']
#'SECTOR' - list of symbols ['Sec1', 'Sec2']
#'EXCHANGE' - list of symbols ['Exch1', 'Exch2']
def subsample_data(filename_data, filename_symbology, dir_pickle, start_date, end_date, query_attribute, query_criteria, include_avg):
    query_criteria_filename = '-'.join(query_criteria[:3])
    pickle_name = dir_pickle+'pickle_sentiment_'+start_date+'_'+end_date+'_'+query_attribute+'_'+query_criteria_filename+'_'+str(include_avg)+'.p'
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
        data = data[(data['DATE'] >= start_date) & (data['DATE'] <= end_date)]
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
    pickle_name = dir_pickle+'pickle_finance_'+source+'_'+start_date+'_'+end_date+'_'+symbols_filename+'.p'
    try: 
        data_finance = pd.read_pickle(pickle_name)
        print("Loaded from pre-created pickle")
    except:
        # print("Scraping and saving data from "+source)
        # get finance data using pandas data reader
        # create df from first symbol
        try:
            print symbols[0]
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
            print symbol
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
    # log bullishness (bull - bear volume)
    df['LOG_BULLISHNESS'] = np.log(1 + df['BULL_SCORED_MESSAGES']) - np.log(1 + df['BEAR_SCORED_MESSAGES'])
    # log bull bear ratio
    df['LOG_BULL_BEAR_RATIO'] = np.log(df['BULL_SCORED_MESSAGES']) / np.log(df['BEAR_SCORED_MESSAGES'])
    # log bullishness bearishness change
    df['LOG_BULL_MINUS_BEAR_CHANGE'] = np.log(1+df['BULL_MINUS_BEAR']).diff()
    # TISf
    df['TISf'] = (1+df['BULL_SCORED_MESSAGES'])/(1+ df['BULL_SCORED_MESSAGES']+df['BEAR_SCORED_MESSAGES'])
    # RTISf
    df['RTISf'] = ((1+df['BULL_SCORED_MESSAGES'])/(1+ df['BULL_SCORED_MESSAGES']+df['BEAR_SCORED_MESSAGES'])).pct_change()
    # diff in total scanned messages
    df['TOTAL_SCANNED_MESSAGES_DIFF'] = df['TOTAL_SCANNED_MESSAGES'].diff()
    # diff in total scanned messages
    df['TOTAL_SENTIMENT_MESSAGES_DIFF'] = (df['BULL_SCORED_MESSAGES']+df['BEAR_SCORED_MESSAGES']).diff()

    replace_nan_num_cols(df)
    pca_cols = df.select_dtypes(include=[np.float, np.int]).columns
    df['PCA_SENTIMENT'] = PCA(n_components=1).fit_transform(df[pca_cols])
    return df

#preprocess finance data with additional columns for:
# log_return, volatility, log_volume_diff
def preprocess_data_finance(df):
    # log return
    df['LOG_RETURN'] = np.log(1 + df['CLOSE'].pct_change())
    # volatitility
    df['VOLATILITY'] = df['HIGH'] - df['LOW']
    # difference in volume
    df['LOG_VOLUME_DIFF'] = np.log(df['VOLUME'].diff())

    replace_nan_num_cols(df)
    pca_cols = df.select_dtypes(include=[np.float, np.int]).columns
    df['PCA_FINANCE'] = PCA(n_components=1).fit_transform(df[pca_cols])

    return df

# take sentiment and finance datasets and apply necessary
# preprocessing per symbol
def preprocess_per_symbol(df_s, df_f):
    return [df_s.groupby('SYMBOL').apply(preprocess_data_sentiment), df_f.groupby('SYMBOL').apply(preprocess_data_finance)]

# build map of columns with nan values
def build_nan_col_list(df):
    nan_col_list = []
    for col in df.columns:
        if df[col].isnull().values.any():
            nan_col_list.append(col)

    return nan_col_list

# function to replace columns with nan valeus with 0 
def replace_nan_num_cols(df):
    num_cols = df.select_dtypes(include=[np.float, np.int]).columns
    lst = [np.inf, -np.inf]
    to_replace = dict((v, lst) for v in num_cols)
    df.replace(to_replace, np.nan, inplace=True)
    df.update(df[num_cols].fillna(value=0))

# more general split-apply-combine
def split_apply_combine(df, key, func, *args):
    # print "args:", args
    return df.groupby(key).apply(func, *args)

# merge sentiment and finance data
# usually called with F, F, T
def merge_sentiment_finance(data_sentiment, data_finance, with_symbol, fill_finance, fill_sentiment):
    if with_symbol:
#         return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='left')
        if fill_finance and fill_sentiment:
            return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='inner')
        if fill_finance:
            return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='left')
        if fill_sentiment:
            return pd.merge(data_sentiment, data_finance, on=['DATE', 'SYMBOL'], how='right')
    else:
        if fill_finance and fill_sentiment:
            return pd.merge(data_sentiment, data_finance, on=['DATE'], how='inner')
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
            print index, "error" 

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

# check adf test
def adf_test(df):
    df_num = df.select_dtypes(include=[np.float, np.int])
    adf_test = pd.DataFrame(index = df_num.columns, columns = ['ADF'])
    for i, column in enumerate(df_num.columns):
        try:
            adf_test.ix[i,0] = sm.tsa.stattools.adfuller(df[column][1:])
        except:
            print "errror"
    return adf_test

# apply rollling window to data
def apply_rolling_window(df, width):
    df = df.ix[1:, :]
    df_num = df.select_dtypes(include=[np.float, np.int])
    df_non_num = df.select_dtypes(exclude=[np.float, np.int])
    df_num_window = pd.rolling_mean(df_num, width, min_periods=1)
    df_window = pd.concat([df_non_num, df_num_window], axis=1)
    return df_window

def correlation_analysis(df, threshold, data_sentiment, data_finance):
    df_num = df.select_dtypes(include=[np.float, np.int])
    res_corr = pd.DataFrame(index = df_num.columns, columns = df_num.columns)
    corr_list = []
    for i, col_i in enumerate(df_num.columns):
        for j, col_j in enumerate(df_num.columns):
            corr = df[col_i].corr(df[col_j])
            res_corr.ix[i,j] = corr
            if abs(corr) > threshold and abs(corr) < 0.99 and i < j and col_i in data_sentiment.columns and col_j in data_finance.columns:
                corr_list.append([col_i, col_j, corr])
    corr_list = list(reversed(sorted(corr_list, key=itemgetter(2))))
    pprint.pprint(corr_list)
    return res_corr

# Sturges formula: k = log_2 n + 1  
def sturges_bin(df):
    n = len(df.index)
    return math.ceil(np.log2(n)+1)
# Rice Rule k = 2 n^{1/3}
def rice_bin(df):
    n = df.count()[1]
    return math.ceil(2*n**(1/3)) 
# Doanes formula for non-normal data.
def doane_bin(data):
    n = data.count()
    # print "doane", n
    if n == 0 or n == 1:
        return 1
    else:
        std = np.std(data)
        g_1 = abs( s.moment(data,3) / s.moment(data, 2))
        std_g_1 = Decimal(6 * (n - 2)) / Decimal( (n + 1) * (n + 2) )
        std_g_1 = math.sqrt(std_g_1)
        bins = round(1 + np.log2(n) + np.log2(1+g_1/std_g_1))
    # debug 
    # print "n ", n, " std ", std, " g_1 ", g_1, " std_g_1 ", std_g_1, " bins "
    return bins

def calc_mutual_information(x, y, bins):
    # print "len(x)", len(x)

    # print "x NaN", x[x == np.nan]
    # print "y NaN", y[y == np.nan]

    try:
        if bins == -1:
            bins = doane_bin(x)
        if bins == np.inf:
            bins = sturges_bin(x)
    except ValueError:
        bins = 10.0
    # print "bins", bins
    try:
        c_xy = np.histogram2d(x, y, bins)[0]
        mi = metrics.mutual_info_score(None, None, contingency=c_xy)
        # print "success"
    except Exception,e: 
        print "error with mi calc", str(e)
        mi = 0
    return mi

def information_surplus(df, time_shift, varx, vary, bins, exante):
    # print df.SYMBOL.unique(), "exante ", exante

    output = []

    if exante:
        shift_range = range(0, -(time_shift+1), -1)
    else:
        shift_range = range(0, time_shift+1)

    # print "len(df.index)", len(df.index)

    for i in shift_range:
        if abs(i) > len(df.index):
            break

        shift_x = df[varx].shift(i)
        # print "shift_x length", len(shift_x)

        mi = 0.0

        if exante:
            end_index = (len(shift_x.index) - 1 - abs(i))
            x = shift_x[1:end_index]
            y = df[vary][1:end_index]
            # print "len(x.index)", len(x.index), "len(x)", len(x), "end_index", end_index
        else:
            # print "exec"
            x = shift_x[1+abs(i):]
            y = df[vary][1+abs(i):]

        mi = calc_mutual_information(x, y, bins)

        if i == 0:
            mi_origin = mi

        if mi_origin == 0: 
            inf_surp_pct = 0
        else:
            inf_surp_pct = (mi - mi_origin) / mi_origin * 100

        output.append({'SHIFT': i, 'MUTUAL_INFORMATION': mi, 'INFORMATION_SURPLUS_DIFF': mi - mi_origin, 'INFORMATION_SURPLUS_PCT': inf_surp_pct})

    output_frame = pd.DataFrame(output)
    return output_frame

# calc net information surplus
def net_information_surplus(df, time_shift, varx, vary, bins):
    # calc mi for ex-ante
    mi_res = information_surplus(df, time_shift, varx, vary, bins, True)
    mi_res_valid = information_surplus(df, time_shift, varx, vary, bins, False)
    # build index of 0 values
    # mi_res['MUTUAL_INFORMATION'][mi_res['MUTUAL_INFORMATION'] <= 0] = 0
    # mi_res_valid['INFORMATION_SURPLUS_PCT'][mi_res_valid['INFORMATION_SURPLUS_PCT'] <= 0] = 0

    mi_res_net = mi_res['MUTUAL_INFORMATION'] > mi_res_valid['MUTUAL_INFORMATION']
    # print mi_res_net.head()
    net_pos_index = mi_res_net[mi_res_net == True].index.tolist()

    zero_list = [x for x in mi_res.index.tolist() if x not in net_pos_index]

    mi_res.ix[zero_list, 'INFORMATION_SURPLUS_PCT'] = 0
    mi_res['INFORMATION_SURPLUS_PCT'][mi_res['INFORMATION_SURPLUS_PCT'] <= 0] = 0

    # print net_pos_index[:10]
    return mi_res

def constrain_mi_res(df):
    idx = df[~(df['INFORMATION_SURPLUS_PCT'] <= 0).values].index.get_level_values('SYMBOL').unique()
    return df.copy().loc[(idx, slice(None)),:]

# save mi results
def save_information_surplus(dir_pickle, df, symbol, start_date, end_date, time_shift, varx, vary, bins, exante, window_size):
    if window_size > 0: 
        pickle_name = dir_pickle+'info_surp_res/'+symbol+'_'+start_date+'_'+end_date+'_'+str(window_size)+'_'+varx+'_'+vary+'_'+str(bins)+'_'+str(exante)+'.p' 
    else:
        pickle_name = dir_pickle+'info_surp_res/'+symbol+'_'+start_date+'_'+end_date+'_'+varx+'_'+vary+'_'+str(bins)+'_'+str(exante)+'.p'
    df.to_pickle(pickle_name)
    print("Saved to pickle: " + pickle_name)

def load_information_surplus(dir_pickle, symbol, start_date, end_date, time_shift, varx, vary, bins, exante, window_size):
    if window_size > 0: 
        pickle_name = dir_pickle+'info_surp_res/'+symbol+'_'+start_date+'_'+end_date+'_'+str(window_size)+'_'+varx+'_'+vary+'_'+str(bins)+'_'+str(exante)+'.p' 
    else:
        pickle_name = dir_pickle+'info_surp_res/'+symbol+'_'+start_date+'_'+end_date+'_'+varx+'_'+vary+'_'+str(bins)+'_'+str(exante)+'.p'
    print("Load pickle: " + pickle_name)
    return pd.read_pickle(pickle_name)
                
# normal pmi func
def pmi_func(df, x, y):
    freq_x = df.groupby(x).transform('count')
    freq_y = df.groupby(y).transform('count')
    freq_x_y = df.groupby([x, y]).transform('count')
    df['pmi'] = np.log( len(df.index) *  (freq_x_y / (freq_x * freq_y)) )
    
# kde pmi func
def kernel_pmi_func(df, x, y, i, b=1.0):
    x = np.array(df[x])
    y = np.array(df[y])
    x_y = np.stack((x, y), axis=-1)
    
    kde_x = KernelDensity(kernel='gaussian', bandwidth=b).fit(x[:, np.newaxis])
    kde_y = KernelDensity(kernel='gaussian', bandwidth=b).fit(y[:, np.newaxis])
    kde_x_y = KernelDensity(kernel='gaussian', bandwidth=b).fit(x_y)
    
    p_x = pd.Series(np.exp(kde_x.score_samples(x[:, np.newaxis])))
    p_y = pd.Series(np.exp(kde_y.score_samples(y[:, np.newaxis])))
    p_x_y = pd.Series(np.exp(kde_x_y.score_samples(x_y)))   
    
    df['PMI_'+str(i)] = np.log( p_x_y / (p_x * p_y) )

def daily_pmi_info_surplus(df, time_shift, varx, vary, exante, bandwidth=1.0):  
    df_copy = df.copy()

    if exante:
        shift_range = range(0, -(time_shift+1), -1)
    else:
        shift_range = range(0, time_shift+1)

    for i in shift_range:
        df_copy['shift_'+varx+'_'+str(i)] = df_copy[varx].shift(i)
        
    if exante:
        df_copy.drop(df.tail(time_shift).index, inplace=True)
        df_copy.drop(df.head(time_shift).index, inplace=True)
    else:
        df_copy.drop(df.head(time_shift).index, inplace=True)
        df_copy.drop(df.tail(time_shift).index, inplace=True)
    
    for i in shift_range:
        kernel_pmi_func(df_copy, 'shift_'+varx+'_'+str(i), vary, i, bandwidth)
        if i == 0:
            df_copy['pmi_is_'+str(i)] = 0.0
        else:
            df_copy['pmi_is_'+str(i)] = (df_copy['PMI_'+str(i)] - df_copy['PMI_'+str(0)]) / df_copy['PMI_'+str(0)] * 100
            # print df_copy['pmi_is_'+str(i)].values

    # print df_copy.head()

    return df_copy

# net daily pmi info surplus
def net_daily_pmi_info_surplus(df, time_shift, varx, vary):
    # add results of pmi to new df
    df_pmi_res = split_apply_combine(df, ['SYMBOL'], daily_pmi_info_surplus, time_shift, 'PCA_SENTIMENT', 'PCA_FINANCE', True)

    df_pmi_res_valid = split_apply_combine(df, ['SYMBOL'], daily_pmi_info_surplus, time_shift, 'PCA_SENTIMENT', 'PCA_FINANCE', False)

    df_pmi_merge = pd.merge(df_pmi_res, df_pmi_res_valid)

    for i in range(1, time_shift):

        pmi_res_net = df_pmi_merge['pmi_is_'+str(-i)] > df_pmi_merge['pmi_is_-'+str(i)]
        net_pos_index = pmi_res_net[pmi_res_net == True].index.tolist()
        zero_list = [x for x in df_pmi_merge.index.tolist() if x not in net_pos_index]

        df_pmi_merge['pmi_is_-'+str(i)] = 0
        df_pmi_merge['pmi_is_-'+str(i)][df_pmi_merge['pmi_is_-'+str(i)] <= 0] = 0

        

        

        # df_pmi_merge.ix[zero_list, 'INFORMATION_SURPLUS_PCT'] = 0
        # df_pmi_merge['INFORMATION_SURPLUS_PCT'][mi_res['INFORMATION_SURPLUS_PCT'] <= 0] = 0




    return df_pmi_merge



# calc net information surplus
# def net_information_surplus(df, time_shift, varx, vary, bins):
#     # calc mi for ex-ante
#     mi_res = information_surplus(df, time_shift, varx, vary, bins, True)
#     mi_res_valid = information_surplus(df, time_shift, varx, vary, bins, False)
#     # build index of 0 values
#     mi_res['INFORMATION_SURPLUS_PCT'][mi_res['INFORMATION_SURPLUS_PCT'] <= 0] = 0
#     mi_res_valid['INFORMATION_SURPLUS_PCT'][mi_res_valid['INFORMATION_SURPLUS_PCT'] <= 0] = 0

#     mi_res_net = mi_res > mi_res_valid
#     net_pos_index = mi_res_net[mi_res_net['INFORMATION_SURPLUS_PCT'] == True].index.tolist()

#     zero_list = [x for x in mi_res.index.tolist() if x not in net_pos_index]

#     mi_res.ix[zero_list, 'INFORMATION_SURPLUS_PCT'] = 0

#     # print net_pos_index[:10]
#     return mi_res

# def constrain_mi_res(df):
    # idx = df[~(df['INFORMATION_SURPLUS_PCT'] <= 0).values].index.get_level_values('SYMBOL').unique()
    # return df.copy().loc[(idx, slice(None)),:]
    # # return mi_res_constrained


def kmeans(df, features):
    df_num = df.select_dtypes(include=[np.float, np.int])
    # print df_num.info()
    df_num = df_num[features]
    # Convert DataFrame to matrix
    mat = df_num.as_matrix()
    # Using sklearn
    km = KMeans()
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    # results = pandas.DataFrame([dataset.index,labels]).T

    data = mat

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.get_cmap('Set3'),
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on sentiment and finance metrics')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

# graphical library function

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    # fig, ax = plt.subplots(figsize=(size, size))
    # cax = ax.matshow(corr, interpolation='nearest')
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    # plt.yticks(range(len(corr.columns)), corr.columns)
    # fig.colorbar(cax)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(size, size))

    # Draw the heatmap using seaborn
    sns.heatmap(corr, vmax=1.0, square=True)

    # Use matplotlib directly to emphasize known networks
    # networks = corrmat.columns.get_level_values("network")
    # for i, network in enumerate(networks):
    #     if i and network != networks[i - 1]:
    #         ax.axhline(len(networks) - i, c="w")
    #         ax.axvline(i, c="w")
    # f.tight_layout()

    # 

    # fig = plt.figure()
    # data_nasdaq_top_100_mkt_cap_symbology_corr = data_nasdaq_top_100_mkt_cap_symbology.corr()
    # # plt.matshow(data_nasdaq_top_100_mkt_cap_symbology_corr)
    # # plt.colorbar(data_nasdaq_top_100_mkt_cap_symbology_corr)

    # labels = data_nasdaq_top_100_mkt_cap_symbology_corr.columns
    # # print labels
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(data_nasdaq_top_100_mkt_cap_symbology_corr, interpolation='nearest')
    # fig.colorbar(cax)

def plot_clustermap(df):
    # corr = df.corr()
    # yticks = corr.index
    
    # sns.clustermap(corr, 'yticklabels=yticks')
    cg=sns.clustermap(df.corr())
    # plt.yticks(rotation=0)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)    
    # plt.show()

def plot_pdf(df):
    df_num = df.select_dtypes(include=[np.float, np.int])

    # rows = df_num / 3

    # f, axes = plt.subplots(3, rows + 1)

    # print axes

    for index in df_num.columns:
        try:
            sns.distplot(df_num[index], color="m")
        except:
            print index, "error (probably Nan)"

def plot_scatter_regression(df, x_name, y_name):
    df.plot(kind='scatter', x=x_name, y=y_name)
    x = df[x_name]
    y = df[y_name]
    idx = np.isfinite(x) & np.isfinite(y)
    m, b = np.polyfit(x[idx], y[idx], 1)
    plt.plot(x, m*x + b, '-')

def plot_info_surplus(results, legend):

    axes = plt.gca()
    ymax = 0

    for i, res in enumerate(results):
        plt.plot(res['SHIFT'], res['INFORMATION_SURPLUS_PCT'])
        ymax = max(res['INFORMATION_SURPLUS_PCT'].max(), ymax)

    axes.set_ylim([0.0,ymax])
    plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Time-shift of sentiment data (days) with financial data')
    plt.ylabel('Information Surplus %')
    plt.show()

def plot_inf_res(df, symbols=[]):

    if len(symbols) > 0:
        df = df.loc[symbols]
    # figsize=(15, 5)
    ax = df.unstack(0).plot(x='SHIFT', y='INFORMATION_SURPLUS_PCT', legend=df.index.levels)
    # plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel('Time-shift of sentiment data (days) with financial data')
    ax.set_ylabel('Information Surplus %')

def plot_lead_trail_res(df_ante, df_post, symbols=[]):

    if len(symbols) < 1:
        print "Try again with a symbol list. (Time constraints)"
    else:
        df_ante = df_ante.loc[symbols]
        df_post = df_post.loc[symbols]

        df_ante.index.set_levels([[str(x)+'_ex-ante' for x in df_ante.index.levels[0]],df_ante.index.levels[1]], inplace=True)
        df_post.index.set_levels([[str(x)+'_ex-post' for x in df_post.index.levels[0]],df_post.index.levels[1]], inplace=True)

        df_merge = pd.concat([df_ante, df_post])
        df_merge['SHIFT'] = abs(df_merge['SHIFT'])

        # print df_merge.index.levels[0].values

        df_merge.unstack(0).plot(x='SHIFT', y='MUTUAL_INFORMATION', legend=[df_merge.index.levels[0].values])



# getting min and max from info surplus

# plt.plot(a_sent_0['SHIFT'], a_sent_0['INFORMATION_SURPLUS_PCT'])
# plt.plot(g_sent_0['SHIFT'], g_sent_0['INFORMATION_SURPLUS_PCT'])


# ymin = min(a_sent_0['INFORMATION_SURPLUS_PCT'].min(), g_sent_0['INFORMATION_SURPLUS_PCT'].min())
# ymin = 0
# ymax = max(a_sent_0['INFORMATION_SURPLUS_PCT'].max(), g_sent_0['INFORMATION_SURPLUS_PCT'].max())


# axes.set_xlim([xmin,xmax])


