#!/usr/bin/env python3

import zipfile
import pandas as pd

# paths to zipped files
bitstamp_zip = 'supervised_learning/time_series/data_files/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv.zip'
coinbase_zip = 'supervised_learning/time_series/data_files/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip'

# unziping the file
with zipfile.ZipFile(bitstamp_zip, 'r') as zip_ref:
    zip_ref.extractall('supervised_learning/time_series/data_files')
with zipfile.ZipFile(coinbase_zip, 'r') as zip_ref:
    zip_ref.extractall('supervised_learning/time_series/data_files')

# loading data
bitstamp_raw_data = pd.read_csv('supervised_learning/time_series/data_files/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
coinbase_raw_data = pd.read_csv('supervised_learning/time_series/data_files/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

# checking if loaded correctly
print("Bitstamp Data Loaded:")
print(bitstamp_raw_data.head())
print("Shape:", bitstamp_raw_data.shape)
print("\nCoinbase Data Loaded:")
print(coinbase_raw_data.head())
print("Shape:", coinbase_raw_data.shape)