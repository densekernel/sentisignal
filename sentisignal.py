''' sentisignal.py
A package for performing cluster analysis of socially informed financial volatility.
'''


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
from sklearn.manifold import TSNE
from radar import radar_graph

# date plot config
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')


# set seaborn styles
sns.set_style("white")
sns.set_context("notebook")


def subsample_data(filename_data, filename_symbology, dir_pickle, start_date, end_date, query_attribute, query_criteria, include_avg):
    ''' subsample_data:
    function to subsample the large data sets for:
    specific date range: start_data - end_date ('YYYY-MM-DD')
    filters : 'SYMBOL' - list of symbols ['SYM1', SYM2']
    'SECTOR' - list of symbols ['Sec1', 'Sec2']
    'EXCHANGE' - list of symbols ['Exch1', 'Exch2']
    '''
    query_criteria_filename = '-'.join(query_criteria[:3])
    pickle_name = dir_pickle+'pickle_sentiment_'+start_date+'_'+end_date+'_'+query_attribute+'_'+query_criteria_filename+'_'+str(include_avg)+'.p'
    try: 
        data = pd.read_pickle(pickle_name)
        print("Loaded from pre-created pickle")

    except:
        print("Subsampling data from csv")
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

    return data


def get_data_finance(source, symbols, start_date, end_date, dir_pickle, sum_data_frame, sum_symbol):
    '''get_data_finance
    function to return historical finance data
    takes a list of ticker symbols
    if sum_data_frame is true -> create a summed value to form an indic
    else create a concatenated dataframe
    '''
    symbols_filename = '-'.join(symbols[:3])
    pickle_name = dir_pickle+'pickle_finance_'+source+'_'+start_date+'_'+end_date+'_'+symbols_filename+'.p'
    try: 
        data_finance = pd.read_pickle(pickle_name)
        print("Loaded from pre-created pickle")

    except:
        # get finance data using pandas data reader
        try:
            data_finance = data.DataReader(symbols[0], source, start_date, end_date)
            data_finance.columns = [x.upper() for x in data_finance.columns]
            if sum_data_frame:
                data_finance['SYMBOL'] = sum_symbol
            else:
                data_finance['SYMBOL'] = symbols[0]
        except: 
            print("Unable to retrieve data for: " + symbols[0])

        # loop through remaining symbols and either add or concatenate
        for symbol in symbols[1:]:
            print symbol
            try:
                symbol_finance = data.DataReader(symbol, source, start_date, end_date)
                # convert headers to uppercase for ease of use
                symbol_finance.columns = [x.upper() for x in symbol_finance.columns]
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
        data_finance.columns = [x.upper() for x in data_finance.columns]
        # Set Date as index
        data_finance = data_finance.sort_values(['DATE'], ascending=[True])
        data_finance.to_pickle(pickle_name)
        
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
    df['PCA_SENTIMENT_CHANGE'] = PCA(n_components=1).fit_transform(df[['LOG_BEAR_RETURN', 'LOG_BULL_RETURN', 'LOG_BULLISHNESS', 'LOG_BULL_BEAR_RATIO', 'LOG_BULL_MINUS_BEAR_CHANGE', 'TOTAL_SENTIMENT_MESSAGES_DIFF', 'TOTAL_SENTIMENT_MESSAGES_DIFF']])
    return df

#preprocess finance data with additional columns for:
# log_return, volatility, log_volume_diff
def preprocess_data_finance(df):
    # log return
    df['LOG_RETURN'] = np.log(1 + df['CLOSE'].pct_change())
    # volatitility
    df['VOLATILITY'] = df['HIGH'] - df['LOW']
    df['LOG_VOLATILITY_DIFF'] = np.log(df['VOLATILITY'].diff())
    # difference in volume
    df['LOG_VOLUME_DIFF'] = np.log(df['VOLUME'].diff())

    replace_nan_num_cols(df)
    pca_cols = df.select_dtypes(include=[np.float, np.int]).columns
    df['PCA_FINANCE'] = PCA(n_components=1).fit_transform(df[pca_cols])
    df['PCA_FINANCE_CHANGE'] = PCA(n_components=1).fit_transform(df[['LOG_RETURN', 'LOG_VOLATILITY_DIFF', 'LOG_VOLUME_DIFF']])

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
    print "length prior to constrain", len(df.index.levels[0])
    idx = df[~(df['INFORMATION_SURPLUS_PCT'] <= 0).values].index.get_level_values('SYMBOL').unique()
    print "length after to constrain", len(idx)
    return df.copy().loc[(idx, slice(None)),:]

def test_mi_significant (df, varsenti, varfinan, bins, test_times):
    mi_res = calc_mutual_information(df[varsenti], df[varfinan],bins)
    not_pass_times = 0
    output = []
    df.index = arange(len(df))
    for simulation in arange(test_times):
        random_pos = random.sample(range(len(df)),len(df))
        df_valid = df.ix[random_pos]
        mi_valid = calc_mutual_information(df_valid[varsenti], df_valid[varfinan],bins)
        if mi_valid > mi_res and not_pass_times < test_times*0.07:
            not_pass_times = not_pass_times + 1

    if not_pass_times >= test_times*0.06:
        mi_res = 0

    output.append({'MUTUAL_INFORMATION': mi_res })
    output_frame = pd.DataFrame(output,columns = ['MUTUAL_INFORMATION'])
    return output_frame
#     return output

def test_sig(grp, df, varx, vary):
    print grp.name
    random_mi_res_list = []
    df_g = df
    df_g.loc[df['SYMBOL'] == grp.name]
    x = df_g[varx]
    y = df_g[vary]
    test_len = 100
    for i in range(0, test_len):
        x.reindex(np.random.permutation(x.index))
        random_mi_res_list.append(calc_mutual_information(x, y, -1))
    j = random_mi_res_list
    pct_val_list = []
    for z in range(grp.index[0][1], grp.index[0][1]+len(grp.index)):
        pct_val = sum(i < grp.ix[z,'MUTUAL_INFORMATION'] for i in j) / test_len * 100
        pct_val_list.append(pct_val)
    grp.loc[:,'TEST_PCT'] = pct_val_list
    return grp

def constrain_test_significant(df, df_origin, varx, vary):
    # print "length prior to test", len(df.index.levels[0])
    df_mi = df.groupby(level=0).apply(test_sig, df_origin, varx, vary)
    idx = df_mi[~(df_mi['TEST_PCT'] <= 95).values].index.get_level_values('SYMBOL').unique()
    print "length after to test", len(idx)
    return df_mi.loc[(idx, slice(None)),:]

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
    
    p_x = np.exp(kde_x.score_samples(x[:, np.newaxis]))
    p_y = np.exp(kde_y.score_samples(y[:, np.newaxis]))
    p_x_y = np.exp(kde_x_y.score_samples(x_y))
    
    # df['PMI_'+str(i)] = np.log( p_x_y / (p_x * p_y) )

    # print "len p_x", len(p_x), "len p_y", len(p_y), "len p x y", len(p_x_y)

    # return df
    vals = np.log(p_x_y / (p_x * p_y))
    # print vals[1]
    return vals


def add_shift_col(df, time_shift, varx, exante):
    if exante:
        shift_range = range(0, -(time_shift+1), -1)
    else:
        shift_range = range(0, time_shift+1)

    for i in shift_range:
        df.loc[:,'shift_'+varx+'_'+str(i)] = df.loc[:,varx].shift(i)

    start = df.index[0] + time_shift
    end = df.index[len(df.index)-1] - time_shift - 1

    return df.ix[start:end,:]

def add_shift_data(df, time_shift, varx):
    df_pmi_res = add_shift_col(df, time_shift, varx, True)
    df_pmi_res_valid = add_shift_col(df, time_shift, varx, False)
    df_pmi_merge = pd.merge(df_pmi_res, df_pmi_res_valid)

    return df_pmi_merge

def daily_pmi_info_surplus(df, time_shift, varx, vary, exante, bandwidth=1.0):  
    # df = df.copy()

    # print df_copy.head(1)

    if exante:
        shift_range = range(0, -(time_shift+1), -1)
    else:
        shift_range = range(0, time_shift+1)

    # for i in shift_range:
    #     df.loc[:,'shift_'+varx+'_'+str(i)] = df[varx].shift(i)
        # df.loc[:,'shift_'+varx+'_'+str(i)] = df.loc[:,'shift_'+varx+'_'+str(i)].fillna(df.loc[:,'shift_'+varx+'_'+str(i)].mean())
        
    # if exante:
        # df = df.drop(df.tail(time_shift).index)
        # df = df.drop(df.head(time_shift).index)
        # end_index = (len(shift_x.index) - 1 - abs(i))
        
    # else:
        # df = df.drop(df.head(time_shift).index)
        # df = df.drop(df.tail(time_shift).index)
    # print "start, end ", (time_shift+1), " ", (len(df)-time_shift-1)
    # print "abs", , " ", 
    # start = df.index[0] + time_shift + 5
    # end = df.index[len(df.index)-1] - time_shift - 1 - 5
    # df.iloc[start:end, :]
    # print "len loc", len(df)

    # df.fillna(df.loc[:,'shift_'+varx+'_'+str(i)], inplace=True)
    
    # print df.head(1)
        # y = df[vary][1:end_index] 
    
    for i in shift_range:
        # df.loc[:,'PMI_'+str(i)] = kernel_pmi_func(df, 'shift_'+varx+'_'+str(i), vary, i, bandwidth)
        df['PMI_'+str(i)] = kernel_pmi_func(df, 'shift_'+varx+'_'+str(i), vary, i, bandwidth)
        # print df['PMI_'+str(i)].head(1)
        if i == 0:
            # j = np.random.random()
            # df.loc[:,('pmi_is_'+str(i))] = 0
            # df.loc[:,('pmi_is_'+str(i))] = 0
            df['pmi_is_'+str(i)] = 0
        else:
            # print i 
            # j = np.random.random()
            # df.loc[:,('pmi_is_'+str(i))] = i + j
            # df.loc[:, 'pmi_is_'+str(i)] = (df['PMI_'+str(i)] - df['PMI_'+str(0)]) / df['PMI_'+str(0)] * 100
            df['pmi_is_'+str(i)] = (df['PMI_'+str(i)] - df['PMI_'+str(0)]) / df['PMI_'+str(0)] * 100
            # print df['pmi_is_'+str(i)].values

    # print df.head()

    return df

# net daily pmi info surplus
def net_daily_pmi_info_surplus(df, time_shift, varx, vary):

    # df = df.copy()
    # print "net daily pmi"

    # print "len df", len(df) 
    # add results of pmi to new df
    df_pmi_res = daily_pmi_info_surplus(df, time_shift, varx, vary, True)
    
    # df_pmi_merge = pd.merge(df_pmi_res, df_pmi_res_valid)
    # print "len df_pmi_merge", len(df_pmi_merge)

    return df_pmi_res
    

def constrain_daily_pmi(df, time_shift, varx, vary):

    # df_pmi_res = df.copy()

    print "len before constaining", len(df.SYMBOL.unique())

    df_pmi_res = split_apply_combine(df, ['SYMBOL'], daily_validate, time_shift, varx, vary)


    print "len after constaining", len(df_pmi_res.SYMBOL.unique())

    return df_pmi_res

    
def daily_validate(df, time_shift, varx, vary):

    df_pmi_res_valid = daily_pmi_info_surplus(df, time_shift, varx, vary, False)
    df_pmi_res = df.copy()

    for i in range(1, time_shift+1):
        # print i
        pmi_res_net = df_pmi_res['PMI_'+str(-i)] > df_pmi_res_valid['PMI_'+str(i)]
        # print pmi_res_net.head(2)
        # net_pos_index = pmi_res_net[pmi_res_net[]]
        zero_list = pmi_res_net[pmi_res_net == False].index[0] 
        df_pmi_res.ix[zero_list, 'pmi_is_'+str(-i)] = 0
        df_pmi_res['pmi_is_'+str(-i)][df_pmi_res['pmi_is_'+str(-i)] <= 0] = 0
        mean = df_pmi_res['pmi_is_'+str(-i)].mean()
        lim = 3*df_pmi_res['pmi_is_'+str(-i)].std()
        # print lim
        # print mean, " ", lim
        df_pmi_res['pmi_is_'+str(-i)][abs(df_pmi_res['pmi_is_'+str(-i)] - mean) >= (lim)] = 0

    # print "moo"
    # print "0 check", all((df[[col for col in df.columns if 'pmi_is' in col]] <= 0).all())

    return df_pmi_res


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

def prep_df_cluster(df, nasdaq_data, nasdaq_features, merge_data, merge_features):
    avg_df_merge = merge_data.groupby('SYMBOL').mean().reset_index() 
    gb = df.groupby(level=0)
    keys = []
    for g, data in gb:
        keys.append(g)
    output = pd.DataFrame()
    output['SYMBOL'] = keys
    output['MAX_INF_SURP_PCT'] = df.groupby(level=0)['INFORMATION_SURPLUS_PCT'].max().values
    output['MAX_MUTUAL_INFORMATION'] = df.groupby(level=0)['MUTUAL_INFORMATION'].max().values
    output['POS_LAG_COUNT'] = df.groupby(level=0)['INFORMATION_SURPLUS_PCT'].agg({'POS_LAG_COUNT' : lambda ts: (ts > 0).sum()})['POS_LAG_COUNT'].values
    output['OPTIMAL_LAG'] = [x[1] for x in df.groupby(level=0)['MUTUAL_INFORMATION'].idxmax().values]
    # print df.groupby(level=0)['INFORMATION_SURPLUS_PCT'].idxmax()
    # print df.groupby(level=0)['MUTUAL_INFORMATION'].max()
    res = pd.merge(output, nasdaq_data[nasdaq_features], left_on='SYMBOL', right_on='Symbol')
    res = pd.merge(res, avg_df_merge[merge_features], left_on='SYMBOL', right_on='SYMBOL')
    return res

def prep_daily_df_cluster(df):
    # avg_df_merge = merge_data.groupby('SYMBOL').mean().reset_index() 
    # gb = df.groupby('SYMBOL')
    df['max_pmi'] = df[[col for col in df.columns if 'PMI' in col]].max(axis=1)
    df['max_pmi_is'] = df[[col for col in df.columns if 'pmi_is' in col]].max(axis=1)
    df['optimal_lag'] = df[[col for col in df.columns if 'pmi_is' in col]].idxmax(axis=1).apply(lambda x: abs(int(x[-1:])))
    df['pos_lag_count'] = df[[col for col in df.columns if 'pmi_is' in col]].astype(bool).sum(axis=1)

    # df.groupby(level=0)['INFORMATION_SURPLUS_PCT'].agg({'POS_LAG_COUNT' : lambda ts: (ts > 0).sum()})['POS_LAG_COUNT'].values
    # keys = []
    # for g, data in gb:
    #     keys.append(g)
    # output = pd.DataFrame()
    # output['SYMBOL'] = keys
    # output['MAX_INF_SURP_PCT'] = df.groupby(level=0)['INFORMATION_SURPLUS_PCT'].max().values
    # output['MAX_MUTUAL_INFORMATION'] = df.groupby(level=0)['MUTUAL_INFORMATION'].max().values
    # output['OPTIMAL_LAG'] = [x[1] for x in df.groupby(level=0)['MUTUAL_INFORMATION'].idxmax().values]
    # print df.groupby(level=0)['INFORMATION_SURPLUS_PCT'].idxmax()
    # print df.groupby(level=0)['MUTUAL_INFORMATION'].max()
    # res = pd.merge(output, nasdaq_data[nasdaq_features], left_on='SYMBOL', right_on='Symbol')
    # res = pd.merge(res, avg_df_merge[merge_features], left_on='SYMBOL', right_on='SYMBOL')

    return df

    # return res

def kmeans(df, features, attributes, num_clusters=4, num_init=10, plot_pie=False, daily=True,):
    df_num = df.select_dtypes(include=[np.float, np.int])
    df_num_2 = df_num[attributes]
    # print df_num.info()
    df_num = df_num[features]
    # Convert DataFrame to matrix
    mat = df_num.as_matrix()
    # Using sklearn
    km = KMeans(init='k-means++', n_clusters=num_clusters, n_init=num_init)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    # results = pandas.DataFrame([dataset.index,labels]).T

    # data = mat

    fig = plt.figure()
    ax = fig.add_subplot(111)

    X_tsne = TSNE(learning_rate=100).fit_transform(mat)
    cm = plt.cm.get_cmap('jet')
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap=cm)

    plt.colorbar(ticks=range(num_clusters))
    # centroids = km.cluster_centers_
    # print len(centroids)
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    # for i, xy in enumerate(zip(centroids[:, 0], centroids[:, 1])):
    #     print xy
    #     ax.annotate('(%s)' % i, xy=xy, textcoords='data')

    df_num_2['kmeans_labels'] = labels

    gb_mean = df_num_2.groupby(['kmeans_labels'])
    gb_mean = gb_mean.mean()
    # df = gb_mean

    gb_mean_norm = gb_mean.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    gb_mean_norm = gb_mean_norm.clip(lower=0.02, upper=0.98)


    for i in gb_mean_norm.index:
        # print gb_mean_norm.ix[i]
        data = gb_mean_norm.ix[i]
        # print data.columns
        d = data
        # d = data.drop('kmeans_labels')
        # d['MarketCap'] = d['MarketCap']
        labels = attributes
        # print len(labels)
        optimum = np.array(d)
        values = np.array(d)
        # print optimum, values
        # print len(optimum)
        # print max(optimum)
        # optimum = [5, 3, 2, 4, 5, 7, 5, 8, 5]

        radar_graph(labels, values, optimum)

    df['kmeans_labels'] = labels
    gb_pie = df.groupy(['kmeans_labels'])

    if plot_pie:    
        for name, data in gb_pie:
            print name
            cs=plt.cm.jet(np.arange(len(gb_pie))/float(len(gb_pie)))
            data.SECTOR.plot(kind='pie', autopct='%.1f', colors=cs)
            plt.legend(loc='best', bbox_to_anchor=(1.1, 1.05))
            plt.axis('equal')

    return gb_mean

    # reduced_data = PCA(n_components=2).fit_transform(data)
    # kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10)
    # kmeans.fit(data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    # h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    # x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    # y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # # Obtain labels for each point in mesh. Use last trained model.
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(Z, interpolation='nearest',
    #            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #            cmap=plt.get_cmap('Set3'),
    #            aspect='auto', origin='lower')

    # print Z

    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # plt.scatter(centroids[:, 0], centroids[:, 1],
    #             marker='x', s=169, linewidths=3,
    #             color='w', zorder=10)
    # plt.title('K-means clustering on sentiment and finance metrics')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

def plot_tsne(df, features):
    df_num = df.select_dtypes(include=[np.float, np.int])
    # print df_num.info()
    df_num = df_num[features]
    # Convert DataFrame to matrix
    mat = df_num.as_matrix()
    X_tsne = TSNE(learning_rate=100).fit_transform(mat)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
    


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

def plot_inf_res(df, symbols=[], plot_top=0, time_shift=0):

    if len(symbols) > 0:
        df = df.loc[symbols]

    if plot_top > 0:
        idx = df.groupby(level=0)['INFORMATION_SURPLUS_PCT'].max().sort_values(ascending=False).index
        df = df.reindex(index=idx, level=0)[0:(time_shift+1)*plot_top]

    grouped = df.groupby(level=0)
    ax = plt.figure()
    first = True
    for i, group in grouped:
        if first:
            ax = group.plot(x='SHIFT', y='INFORMATION_SURPLUS_PCT', label=str(i))
            first = False
        else:
            group.plot(ax=ax, x='SHIFT', y='INFORMATION_SURPLUS_PCT', label=str(i))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1.0)
    ax.set_xlabel('Time-shift of sentiment data (days) with financial data')
    ax.set_ylabel('Information Surplus %')

def plot_daily_inf_res(df, symbols=[], plot_top=0):
    df = df.copy()
    # data_nasdaq_top_100_preprocessed_merge.groupby('SYMBOL')
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    
    df['max_pmi_is'] = df[[col for col in df.columns if 'pmi_is' in col]].max(axis=1)
    
    if len(symbols) > 0:
        df = df.loc[symbols]

    if plot_top > 0:
        idx = df.groupby('SYMBOL')['max_pmi_is'].max().sort_values(ascending=False).index[:plot_top].values
        print idx
        df = df.loc[list(idx)]
#         df = df.reindex(index=idx)

    fig, ax = plt.subplots(figsize=(15,5))
    for key, grp in df.groupby('SYMBOL'):
        print "key", key
    #     grp.reset_index()
    #     print grp.DATE
        
        ax.plot(grp.DATE.reset_index(drop=True), grp['max_pmi_is'], label=key)
    #     grp['D'] = pd.rolling_mean(grp['B'], window=5)    
    #     plt.plot(grp['D'], label='rolling ({k})'.format(k=key))

    # datemin = (df.DATE.min().year)
    # datemax = (df.DATE.max().year + 1)
    # print datemin, datemax
    # ax.set_xlim(datemin, datemax)
    ax.set_ylim(0, 500)


    plt.legend(loc='best')
    plt.ylabel('PMI IS (-2)')
    fig.autofmt_xdate()
    plt.show()

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

        # df_merge.unstack(0).plot(x='SHIFT', y='MUTUAL_INFORMATION', legend=[df_merge.index.levels[0].values])

        grouped = df_merge.groupby(level=0)
        ax = plt.figure()
        first = True
        for i, group in grouped:
            if first:
                ax = group.plot(x='SHIFT', y='MUTUAL_INFORMATION', label=str(i))
                first = False
            else:
                group.plot(ax=ax, x='SHIFT', y='MUTUAL_INFORMATION', label=str(i))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=1.0)
        ax.set_xlabel('Time-shift of sentiment data (days) with financial data')
        ax.set_ylabel('Information Surplus %')
