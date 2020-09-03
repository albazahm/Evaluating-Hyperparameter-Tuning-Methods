# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import dask
import dask_ml
import dask.dataframe as dd



def load_data(path):
	"""
	Loads the data from path and returns a pandas dataframe

	Args
	path <str> path to the rawfile

	Returns:
	df <dataframe> data_read
	"""
	pandas_data = pd.read_csv(path, header=None)
	print('Loaded dataframe of size: ', pandas_data.shape)

	return pandas_data

def create_test_train(df, row_num, start_col, end_col, save_path_train, save_path_test):
	"""
	Create Test and train set and saves them

	Args
	df <dataframe>: df dataframe
	row_num <int>: row number from where test data starts
	start_col <int>: column number from where we keep features
	end_col <int>: column number up to where we keep features
	save_path_train <str>: path to save train data
	save_path_test <str>: path to save test data
	"""
	train_df = df.iloc[:row_num, start_col:end_col]
	test_df = df.iloc[row_num:, start_col:end_col]
	#dd.from_pandas(train_df, npartitions=3)
	#dd.from_pandas(train_df, npartitions=3)
	train_df.to_csv(save_path_train, index=False)
	test_df.to_csv(save_path_test, index=False)
	return train_df, test_df

def dask_train_test(save_path_train, save_path_test):
	"""
	Create dask dataframe from saved csv files of train and test data

	Args
	path_train <str>: path to the train data
	path_test <str>: path to the test data

	Returns:
	train_df <dataframe>: dask train dataframe
	test_df <dataframe>: dask test dataframe
	"""
	train_df = dd.read_csv(save_path_train)
	test_df = dd.read_csv(save_path_test)

	return train_df, test_df

def concat_data(df_train, df_test):
	"""
	Concatenates train and test dak dataframe

	Args:
	df_train <dataframe>: dask dataframe of train data
	df_test <dataframe>: dask dataframe of test data
	"""
	df = dd.concat([df_train, df_test], axis=0)
	df = df.compute()
	return df
