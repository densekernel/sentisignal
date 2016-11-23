# senti-signal

A package for performing cluster analysis of socially informed financial volatility.

## Project structure

We suggest using the following project structure:

```
/project
  /app
  /data
    /csv (original data)
    /gz (original zip data)
    /output (store results here)
    /pickles (store immediate results as pickles here for quicker access)
    /subsamples (subsamples of original data)
  /py
      sentisignal.py (library containing methods)
      ... (e.g., iPython Notebooks)
```

## Documentation

    
| Function                     | Description                                                                                                                                                                                               |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| subsample_data()             | Subsample sentiment data from the large CSV’s using query of dates, sector, exchange and specific stock symbols and save query to pickle. Reload existing queries from pickle files instead of resampling |
| get_data_finance()           | Scrape Yahoo! finance data for a list of stock symbols for a time period and save to pickle. Reload existing queries from pickle files instead of repeating retrieval                                     |
| preprocess_data_sentiment()  | Process sentiment data with added statistics                                                                                                                                                              |
| preprocess_data_finance()    | Process finance data with added statistics                                                                                                                                                                |
| preprocess_per_symbol()      | Merges pre-processed sentiment and finance data frames and groups instances by the corresponding symbol.                                                                                                  |
| build_nan_col_list()         | Creates map of columns with NaN values                                                                                                                                                                    |
| replace_nan_num_cols()       | Replaces NaN columns with 0                                                                                                                                                                               |
| split_apply_combine()        | Groups data frame by specified key and applies specified function and arguments                                                                                                                           |
| merge_sentiment_finance()    | Merge sentiment and finance data                                                                                                                                                                          |
| check_pdf()                  | Generate probability distribution function of variables in a dataframe and graphically display results                                                                                                    |
| check_acf()                  | Generate auto correlations function of variables in a dataframe and graphically display results                                                                                                           |
| adf_test()                   | Perform adf test on each variable in a DataFrame                                                                                                                                                          |
| apply_rolling_window()       | Convert data into rolling average of a given window size                                                                                                                                                  |
| correlation_analysis()       | Compute pairwise correlation between given variables. Output a matrix of all correlation coefficients.                                                                                                    |
| sturges_bin()                | Calculate bin size using Sturge’s formula                                                                                                                                                                 |
| rice_bin()                   | Calculate bin size using Rice’s formula                                                                                                                                                                   |
| doane_bin()                  | Calculate bin size using Doane’s formula                                                                                                                                                                  |
| calc_mutual_information()    | Computes mutual information between sentiment and finance feature using bin number from one of the three methods above.                                                                                   |
| information_surplus()        | Computes the information gain percentage for each ex-ante time shift for the,variables provided to calculate mutual information.                                                                          |
| net_information_surplus()    | Returns the net information surplus for the computation in the previous method.                                                                                                                           |
| constrain_mi_res()           | Returns the shifts with positive information surplus.                                                                                                                                                     |
| save_information_surplus()   | Saves the results from the MI/IS tests as a pickle that can be accessed locally for further future analyses.                                                                                              |
| test_mi_significant()        | Tests the significance of mutual information for two features used                                                                                                                                        |
| test_sig()                   | Tests if significance is better than randomly permutated data                                                                                                                                             |
| constrain_test_significant() | Tests at significance level of 95%                                                                                                                                                                        |
| save_information_surplus()   | Saves mutual information results                                                                                                                                                                          |
| load_information_surplus()   | Loads the saves results from,save_information_surplus.                                                                                                                                                    |
| pmi_func()                   | Calculates point-wise mutual information (i.e. single event MI rather than average of all events). This provides a more granular MI metric.                                                               |
| kernel_pmi_func()            | Applies an estimated kernel density function tocalculate high dimensional MI using point-wise mutual information.                                                                                         |
| add_shift_col()              | Applies the specified ex-ante time-shift to the data for testings                                                                                                                                         |
| add_shift_data()             | Throughput using add_shift_col, merging respective dataframe with shifted column                                                                                                                          |
| daily_pmi_info_surplus()     | Calculates information surplus using point-wise mutual information.                                                                                                                                       |
| net_daily_pmi_info_surplus() | Returns data frame of information surplus results,using point-wise mutual information using PCA/dimensionality-reduced finance and sentiment features                                                     |
| constrain_daily_pmi()        | Using PMI, prints the validated companies before and after                                                                                                                                                |
| daily_validate()             | Validates PMI surplus using specified sentiment and finance features                                                                                                                                      |
| prep_df_cluster()            | Specify features used for k-means clustering                                                                                                                                                              |
| prep_daily_df_cluster()      | Specify features used for PMI to k-means                                                                                                                                                                  |
| kmeans()                     | k-means clustering on sentiment and finance metrics.                                                                                                                                                      |
| plot_tsne()                  | Scatter plot using TSNE method for reducing dimensions                                                                                                                                                    |
| plot_corr()                  | Creates a heat map of correlation coefficients for each column/feature provided.                                                                                                                          |
| plot_clustermap()            | Creates a hierarchical clustered heat map of correlations. This allows us to identify groups of correlating features.                                                                                     |
| plot_pdf()                   | Plots probability density function.                                                                                                                                                                       |
| plot_scatter_regression()    | Fits a regression line to a general scatterplot.                                                                                                                                                          |
| plot_info_surplus()          | Plots the time-shift mutual information less no time-shift mutual information                                                                                                                             |
| plot_inf_res()               | Plots a curve of information surplus percentage by time-shift. This is used to determine/visualize where and when information surplus can be seen in the data.                                            |
| plot_daily_inf_res()         | Optimal/max PMI analysed against for each company specified                                                                                                                                               |
| plot_lead_trail_res()        | Plot time-shift of results against the information surplus for specified symbols. This gives us a look at what IS is leading and trailing.                                                                |


## Credits

- http://matplotlib.org/examples/api/radar_chart.html