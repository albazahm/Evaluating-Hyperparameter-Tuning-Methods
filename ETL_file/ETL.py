# -*- coding: utf-8 -*-

import pandas as pd
import configparser
import os
import warnings
from sklearn.exceptions import DataConversionWarning
import argparse
import json
import pickle as pickle
from extensions.preprocess import Preprocessor
import extensions.features as features
import extensions.utilities_ETL as utilities_ETL

pd.options.mode.chained_assignment = None
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def _get_args():
    """Get input arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_path",
                        default="data/higgs_data.csv",
                        help="path to Higgs data",
                        type=str)

    parser.add_argument("--row_num",
                        default=10500000,
                        help="row number from which test data starts",
                        type=int)

    parser.add_argument("--result_path",
                        default="clean_data/",
                        help="path to clean data",
                        type=str)

    parser.add_argument("--drop_threshold",
                        default=50,
                        help="Percentage of missing values in a feature below which a column will be dropped",
                        type=int)

    parser.add_argument("--category_threshold",
                        default=1,
                        help="Count of categories below which (or equal) a category will be grouped together as 'Other'",
                        type=int)

    parser.add_argument("--missing_method",
                        default='fill',
                        help="method to handle missing values in a feature",
                        type=str,
                        choices= ['fill','drop'])

    parser.add_argument("--scale_method",
                        default='standardscaler',
                        help="scaling algorithm to use on the data",
                        type=str,
                        choices=['standardscaler','minmax', 'robust'])

    parser.add_argument("--scale_range",
                        default=(0,1),
                        help="Specifies the upper and lower limits of scaling; applicable only for the minmax method only",
                        type=tuple)

    parser.add_argument("--transform_method",
                        default='yeo-johnson',
                        help="transformation algorithm to use on the data",
                        type=str,
                        choices=['yeo-johnson','box-cox'])

    parser.add_argument("--max_depth",
                        default=1,
                        help="maximum allowed depth of integers",
                        type=int)

    parser.add_argument("--cor_threshold",
                        default=0.85,
                        help="number that determines maximum features correlation limit",
                        type=float)

    parser.add_argument("--agg_primitives_list",
                        default=None,
                        help="list of Aggregation Feature types to apply",
                        type=list)

    parser.add_argument("--trans_primitives_list",
                        default=['multiply_numeric', 'add_numeric'],
                        help="list of Transform Feature functions to apply",
                        type=list)

    parser.add_argument("--num_total_features",
                        default=400,
                        help="total number of features to select",
                        type=int)

    return parser.parse_args()

def main():
    """
    The main function, reads all params, reads data, splits data into train
    and test, preprocesses data, and performs feature engineering and selection
    """
    args = _get_args()
    data_path = args.data_path
    row_num = args.row_num
    result_path = args.result_path
    drop_threshold = args.drop_threshold
    category_threshold = args.category_threshold
    missing_method = args.missing_method
    scale_method = args.scale_method
    scale_range = args.scale_range
    transform_method = args.transform_method
    max_depth = args.max_depth
    cor_threshold = args.cor_threshold
    agg_primitives_list = args.agg_primitives_list
    trans_primitives_list = args.trans_primitives_list
    num_total_features = args.num_total_features

    #read config file
    with open('config_ETL.json') as file_object:
        # store file data in object
        config_data = json.load(file_object)
        save_path_train = config_data["save_path_train"]
        save_path_test = config_data["save_path_test"]
        start_col = config_data["start_col"]
        end_col = config_data["end_col"]
        col_names = config_data["col_names"]
        features_list = config_data["features"]
        numeric_features_list = config_data["numeric_features"]
        categoric_features_list = config_data["categoric_features"]
        target = config_data["target"]
        labels = config_data["labels"]

    #read the dataframe
    df_data = utilities_ETL.load_data(data_path)

    #assign column names
    df_data.columns = col_names

    #split into training and testing
    utilities_ETL.create_test_train(df_data, row_num, start_col, end_col, save_path_train, save_path_test )
    train_df, test_df = utilities_ETL.dask_train_test(save_path_train, save_path_test)

    #instantiate the preprocessor class
    x = Preprocessor(numeric_features_list, target, train_df, labels, test_df, categoric_features_list, drop_threshold,
                     category_threshold, missing_method, scale_method, scale_range, transform_method)
    #apply preprocessing on training set
    df_train_processed = x.execute(invalid=True, duplicates=True, missing=True, scale=True, transform=True, encode_target=False, train=True)
    #apply preprocessing on testing set
    df_test_processed = x.execute(invalid=False, duplicates=False, missing=True, scale=True, transform=True, encode_target=False, train=False)

    #append testing set to training set(keep testing set at last 500K rows)
    df_processed = utilities_ETL.concat_data(df_train_processed, df_test_processed)
    os.remove(save_path_train)
    os.remove(save_path_test)
    #instantiate feature engineering class
    y = features.FeatureEng(df_processed, categoric_features_list, target, agg_primitives_list, trans_primitives_list, max_depth, cor_threshold)
    #apply feature engineering
    y.df_with_new_features()
    #instantiate feature selection class
    row_num = df_train_processed.shape[0]
    y_2 = features.FeatureSelect(y.df_new_features, num_total_features, row_num)
    #apply feature selection
    y_2.selected_features_df()
    y_2.df_selected_columns_train.to_csv(result_path + 'train.csv', index=False)
    y_2.df_selected_columns_test.to_csv(result_path + 'test.csv', index=False)

if __name__ == "__main__":
    args = _get_args()
    try:
        os.makedirs(args.result_path)
    except OSError as exc:
        print("Data will be saved in clean_data directory")

    main()



