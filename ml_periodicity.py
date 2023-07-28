#!/usr/bin/python3
"""

Author:Tinc0 CHAN
e-mail: tincochan@foxmail.com, koumar@cesnet.cz

Copyright (C) 2023 CESNET

LICENSE TERMS

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the Company nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

ALTERNATIVELY, provided that this notice is retained in full, this product may be distributed under the terms of the GNU General Public License (GPL) version 2 or later, in which case the provisions of the GPL apply INSTEAD OF those given above.

This software is provided as is'', and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the company or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""
import csv
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import warnings
warnings.filterwarnings('ignore')
import argparse
from argparse import RawTextHelpFormatter

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import seaborn as sns

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# ML algorithms
from xgboost import XGBClassifier

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_absolute_error


periodic_features = [
    "packet_value",
    "packet_value_x",
    "packet_value_y",
    "bytes_value",
    "bytes_value_x",
    "bytes_value_y",
    "duration_value",
    "duration_value_x",
    "duration_value_y",
    "difftimes_value",
    "difftimes_value_x",
    "difftimes_value_y",
]

statistics_features = [
    "packet_mean",
    "packet_std",
    "packet_skewness",
    "packet_kurtosis",
    "bytes_mean",
    "bytes_std",
    "bytes_skewness",
    "bytes_kurtosis",
    "duration_mean",
    "duration_std",
    "duration_skewness",
    "duration_kurtosis",
    "difftimes_mean",
    "difftimes_std",
    "difftimes_skewness",
    "difftimes_kurtosis",
]

frequency_features = [
    "max_power",
    "max_frequency",
    "min_power",
    "min_frequency",
    "spectral_energy",
    "spectral_entropy",
    "spectral_kurtosis",
    "spectral_skewness",
    "spectral_rolloff",
    "spectral_cetroid",
    "spectral_spread",
    "spectral_slope",
    "spectral_crest",
    "spectral_flux",
    "spectral_bandwidth",
]

features = periodic_features + statistics_features + frequency_features

def get_confusion_matrix(y_test, y_pred):
    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_test, y_pred)
    percentage_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    return matrix, percentage_matrix

def plot_confusion_matrix(matrix, percentage_matrix, model=""):
    # Build the plot
    fig, ax = plt.subplots(1, 2,figsize=(15,5))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10}, fmt='g',
                cmap=plt.cm.Greens, linewidths=0.2, ax=ax[0])
    sns.heatmap(percentage_matrix, annot=True, annot_kws={'size':10}, fmt='g',
                cmap=plt.cm.Greens, linewidths=0.2, ax=ax[1])
    # Add labels to the plot
    plt.tick_params(axis='both', which='minor', labelsize=5)
    # tick_marks = np.arange(len(classifications_array)) + 0.5
    # plt.xticks(tick_marks, classifications_array, rotation=85)
    # plt.yticks(tick_marks, classifications_array, rotation=0)
    ax[0].set_xlabel('Predicted label')
    ax[0].set_ylabel('True label')
    ax[0].set_title(f'Absolute Confusion Matrix for {model}')
    ax[1].set_xlabel('Predicted label')
    ax[1].set_ylabel('True label')
    ax[1].set_title(f'Relative Confusion Matrix for {model}')
    # pyplot.savefig("ddos_timeseries_plugin_classification.eps", format="eps")
    plt.show()

def handle_nan(df, label=None):
    if label is None:
        df.loc[df.label == "Other", "label"] = False
        df.loc[df.label == "Miner", "label"] = True
        df['label'] = df.label.astype(bool)
    else:
        df['label'] = label
        df['label'] = df.label.astype(bool)
    df.replace([np.inf], -1, inplace=True)
    df.replace([-np.inf], -1, inplace=True)
    for F in features:
        if F in frequency_features: 
            df.loc[df[F].isnull(), F] = -1
        else:
            df.loc[df[F].isnull(), F] = 0
    return df

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def tunne_together_XGBoost_classification(df_tunne, _features, verbose=True, test_size=0.3):
    X=df_tunne[_features]  # Features
    y=df_tunne['label']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify=y)
    
    def objective(space):
        clf = XGBClassifier(
                        n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']), 
                        # subsample=int(space['subsample']),
                        # eta=space['eta'],
                            )

        evaluation = [( X_train, y_train), ( X_test, y_test)]

        clf.fit(X_train, y_train,
                eval_set=evaluation, eval_metric="auc",
                early_stopping_rounds=10,verbose=False)

        pred = clf.predict(X_test)
        # accuracy = mean_absolute_error(y_test, pred)
        # accuracy = accuracy_score(y_test, pred)
        accuracy = f1_score(y_test, pred)
        if verbose is True:
            print ("SCORE:", accuracy)
        return {'loss': -accuracy, 'status': STATUS_OK }
    
    space={'max_depth': hp.quniform("max_depth", 5, 30, 2),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 0,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 20, 1),
        'n_estimators': hp.quniform('n_estimators', 80, 400, 20),
        # 'subsample': hp.quniform('subsample', 3, 20, 1),
        # 'eta': hp.quniform('eta', 0.005, 0.3, 0.005),
        'seed': 0
    }
    
    trials = Trials()
    
    best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)
        
    print("The best hyperparameters are : ","\n")
    print(best_hyperparams)
    return best_hyperparams

def XGBoost_classification_tunned(df_features, best_hyperparams, test_size=0.3):
      X=df_features[features]  # Features
      y=df_features['label']  # Labels
      #Split on train and test
      X_train_a, X_test_a, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y) # 70% training and 30% test
      X_train = X_train_a[features]
      X_test = X_test_a[features] 
      model = XGBClassifier(
            n_estimators = int(best_hyperparams['n_estimators']), max_depth = int(best_hyperparams['max_depth']), gamma = best_hyperparams['gamma'],
            reg_alpha = int(best_hyperparams['reg_alpha']),min_child_weight=int(best_hyperparams['min_child_weight']),
            colsample_bytree=int(best_hyperparams['colsample_bytree']),
            # subsample=int(best_hyperparams['subsample']),
            # eta=best_hyperparams['eta'],
      )
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      
      accuracy = metrics.accuracy_score(y_test, y_pred)  * 100
      precision = precision_score(y_test, y_pred)  * 100
      recall = recall_score(y_test, y_pred)  * 100
      F1 = f1_score(y_test, y_pred)  * 100
      print("     {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(accuracy,precision,recall,F1))
      matrix, percentage_matrix = get_confusion_matrix(y_test, y_pred)
      # plot_confusion_matrix(matrix, percentage_matrix, model="XGBoost")
      return model

def while_XGBoost_classification_tunned(df_features, best_hyperparams, cycles=100, test_size=0.3):
    X=df_features[features]  # Features
    y=df_features['label']  # Labels
    best_results = (0,0,0,0, None,None, None)
    i = 0
    print(f"\r    {i}", end="")
    for i in range(cycles):
        print(f"\r    {i}", end="", flush=True)
        #Split on train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y) # 70% training and 30% test
        model = XGBClassifier(
                        n_estimators = int(best_hyperparams['n_estimators']), max_depth = int(best_hyperparams['max_depth']), gamma = best_hyperparams['gamma'],
                        reg_alpha = int(best_hyperparams['reg_alpha']),min_child_weight=int(best_hyperparams['min_child_weight']),
                        colsample_bytree=int(best_hyperparams['colsample_bytree']),
                        # subsample=int(best_hyperparams['subsample']),
                        # eta=best_hyperparams['eta'],
                )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)  * 100
        precision = precision_score(y_test, y_pred)  * 100
        recall = recall_score(y_test, y_pred)  * 100
        F1 = f1_score(y_test, y_pred)  * 100
        if best_results[0] <= F1:
            matrix, percentage_matrix = get_confusion_matrix(y_test, y_pred)
            best_results = (accuracy, precision, recall, F1, matrix, percentage_matrix, model)
    print("")
    return best_results

def validate_model(best_model, df_validation):
    if best_model is None:
        return (0, 0, 0, 0, [], [])
    X_validation=df_validation[features]  # Features
    y_validation=df_validation['label']  # Labels
        
    y_pred = best_model.predict(X_validation)
    accuracy = metrics.accuracy_score(y_validation, y_pred)  * 100
    precision = precision_score(y_validation, y_pred)  * 100
    recall = recall_score(y_validation, y_pred)  * 100
    F1 = f1_score(y_validation, y_pred)  * 100                
    
    matrix, percentage_matrix = get_confusion_matrix(y_validation, y_pred)
    # plot_confusion_matrix(matrix, percentage_matrix, model="XGBoost")
    
    feat_importances = pd.Series(best_model.feature_importances_, index = X_validation.columns).to_dict()

    
    return (accuracy, precision, recall, F1, matrix, percentage_matrix, feat_importances)
    
def parse_arguments():
    """Function for set arguments of module.

    Returns:
        argparse: Return setted argument of module.
    """
    parser = argparse.ArgumentParser(
        description="""

    Usage:""",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output_file",
        help="Output file.",
        type=str,
        metavar="NAME.SUFFIX",
        default="test.csv",
    )
    parser.add_argument(
        "--input_file",
        help="INput file mark. Example: '0.01.0.99'.",
        type=str,
        metavar="NAME.SUFFIX",
        default="test.csv",
    )
    parser.add_argument(
        "--validation",
        help="Number of ... .",
        type=float,
        metavar="FLOAT",
        default=0.3,
    )
    parser.add_argument(
        "--test",
        help="Number of ... .",
        type=float,
        metavar="NUMBER",
        default=0.3,
    )
    parser.add_argument(
        "-k",
        help="Number of cycles.",
        type=int,
        metavar="NUMBER",
        default=10,
    )
    arg = parser.parse_args()
    return arg

HEADER = ['INDEX', 'TIME_CUTTING', 'accuracy', 'precision', 'recall', 'F1', 'matrix', 'percentage_matrix', 'accuracy_validation', 'precision_validation', 'recall_validation', 'F1_validation', 'matrix_validation', 'percentage_matrix_validation', 'feature_importance']

def main():
    """Main function of the module."""
    arg = parse_arguments()
    
    df_features_86400 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.86400.{arg.input_file}.csv")
    df_features_43200 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.43200.{arg.input_file}.csv")
    df_features_21600 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.21600.{arg.input_file}.csv")
    df_features_14400 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.14400.{arg.input_file}.csv")
    df_features_7200 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.7200.{arg.input_file}.csv")
    df_features_3600 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.3600.{arg.input_file}.csv")
    df_features_1800 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.1800.{arg.input_file}.csv")
    df_features_900 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.900.{arg.input_file}.csv")
    df_features_600 = pd.read_csv(f"periodicity_features/cryptominers_final_fixed.periodicity_features.600.{arg.input_file}.csv")

    df_validation_86400 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.86400.{arg.input_file}.csv")
    df_validation_43200 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.43200.{arg.input_file}.csv")
    df_validation_21600 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.21600.{arg.input_file}.csv")
    df_validation_14400 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.14400.{arg.input_file}.csv")
    df_validation_7200 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.7200.{arg.input_file}.csv")
    df_validation_3600 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.3600.{arg.input_file}.csv")
    df_validation_1800 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.1800.{arg.input_file}.csv")
    df_validation_900 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.900.{arg.input_file}.csv")
    df_validation_600 = pd.read_csv(f"periodicity_features/evaluation.periodicity_features.600.{arg.input_file}.csv")

    df_features_86400 = handle_nan(df_features_86400)
    df_features_43200 = handle_nan(df_features_43200)
    df_features_21600 = handle_nan(df_features_21600)
    df_features_14400 = handle_nan(df_features_14400)
    df_features_7200 = handle_nan(df_features_7200)
    df_features_3600 = handle_nan(df_features_3600)
    df_features_1800 = handle_nan(df_features_1800)
    df_features_900 = handle_nan(df_features_900)
    df_features_600 = handle_nan(df_features_600)

    df_validation_86400 = handle_nan(df_validation_86400)
    df_validation_43200 = handle_nan(df_validation_43200)
    df_validation_21600 = handle_nan(df_validation_21600)
    df_validation_14400 = handle_nan(df_validation_14400)
    df_validation_7200 = handle_nan(df_validation_7200)
    df_validation_3600 = handle_nan(df_validation_3600)
    df_validation_1800 = handle_nan(df_validation_1800)
    df_validation_900 = handle_nan(df_validation_900)
    df_validation_600 = handle_nan(df_validation_600)
    
    df_features_86400 = df_features_86400[["label"] + features].copy()
    df_features_43200 = df_features_43200[["label"] + features].copy()
    df_features_21600 = df_features_21600[["label"] + features].copy()
    df_features_14400 = df_features_14400[["label"] + features].copy()
    df_features_7200 = df_features_7200[["label"] + features].copy()
    df_features_3600 = df_features_3600[["label"] + features].copy()
    df_features_1800 = df_features_1800[["label"] + features].copy()
    df_features_900 = df_features_900[["label"] + features].copy()
    df_features_600 = df_features_600[["label"] + features].copy()

    df_validation_86400 = df_validation_86400[["label"] + features].copy()
    df_validation_43200 = df_validation_43200[["label"] + features].copy()
    df_validation_21600 = df_validation_21600[["label"] + features].copy()
    df_validation_14400 = df_validation_14400[["label"] + features].copy()
    df_validation_7200 = df_validation_7200[["label"] + features].copy()
    df_validation_3600 = df_validation_3600[["label"] + features].copy()
    df_validation_1800 = df_validation_1800[["label"] + features].copy()
    df_validation_900 = df_validation_900[["label"] + features].copy()
    df_validation_600 = df_validation_600[["label"] + features].copy()

    df_features_86400 = clean_dataset(df_features_86400)
    df_features_43200 = clean_dataset(df_features_43200)
    df_features_21600 = clean_dataset(df_features_21600)
    df_features_14400 = clean_dataset(df_features_14400)
    df_features_7200 = clean_dataset(df_features_7200)
    df_features_3600 = clean_dataset(df_features_3600)
    df_features_1800 = clean_dataset(df_features_1800)
    df_features_900 = clean_dataset(df_features_900)
    df_features_600 = clean_dataset(df_features_600)

    df_validation_86400 = clean_dataset(df_validation_86400)
    df_validation_43200 = clean_dataset(df_validation_43200)
    df_validation_21600 = clean_dataset(df_validation_21600)
    df_validation_14400 = clean_dataset(df_validation_14400)
    df_validation_7200 = clean_dataset(df_validation_7200)
    df_validation_3600 = clean_dataset(df_validation_3600)
    df_validation_1800 = clean_dataset(df_validation_1800)
    df_validation_900 = clean_dataset(df_validation_900)
    df_validation_600 = clean_dataset(df_validation_600)

    with open(arg.output_file, "w") as w:
        writer = csv.writer(w, delimiter=';')
        writer.writerow(HEADER)
        for k in range(arg.k):
            print("######################################################################################################################")
            print(k, "   ######################################################################################################################")
            print("######################################################################################################################")
            print("  hyperparameters tunning")
            best_hyperparams_86400 = tunne_together_XGBoost_classification(df_features_86400, features, verbose=False, test_size=arg.validation)
            best_hyperparams_43200 = tunne_together_XGBoost_classification(df_features_43200, features, verbose=False, test_size=arg.validation)
            best_hyperparams_21600 = tunne_together_XGBoost_classification(df_features_21600, features, verbose=False, test_size=arg.validation)
            best_hyperparams_14400 = tunne_together_XGBoost_classification(df_features_14400, features, verbose=False, test_size=arg.validation)
            best_hyperparams_7200 = tunne_together_XGBoost_classification(df_features_7200, features, verbose=False, test_size=arg.validation)
            best_hyperparams_3600 = tunne_together_XGBoost_classification(df_features_3600, features, verbose=False, test_size=arg.validation)
            best_hyperparams_1800 = tunne_together_XGBoost_classification(df_features_1800, features, verbose=False, test_size=arg.validation)
            best_hyperparams_900 = tunne_together_XGBoost_classification(df_features_900, features, verbose=False, test_size=arg.validation)
            best_hyperparams_600 = tunne_together_XGBoost_classification(df_features_600, features, verbose=False, test_size=arg.validation)

            print("  tune random numbers in models")
            accuracy_86400, precision_86400, recall_86400, F1_86400, matrix_86400, percentage_matrix_86400, best_model_86400 = while_XGBoost_classification_tunned(df_features_86400, best_hyperparams_86400, cycles=10, test_size=arg.validation)
            accuracy_43200, precision_43200, recall_43200, F1_43200, matrix_43200, percentage_matrix_43200, best_model_43200 = while_XGBoost_classification_tunned(df_features_43200, best_hyperparams_43200, cycles=10, test_size=arg.validation)
            accuracy_21600, precision_21600, recall_21600, F1_21600, matrix_21600, percentage_matrix_21600, best_model_21600 = while_XGBoost_classification_tunned(df_features_21600, best_hyperparams_21600, cycles=10, test_size=arg.validation)
            accuracy_14400, precision_14400, recall_14400, F1_14400, matrix_14400, percentage_matrix_14400, best_model_14400 = while_XGBoost_classification_tunned(df_features_14400, best_hyperparams_14400, cycles=10, test_size=arg.validation)
            accuracy_7200, precision_7200, recall_7200, F1_7200, matrix_7200, percentage_matrix_7200, best_model_7200 = while_XGBoost_classification_tunned(df_features_7200, best_hyperparams_7200, cycles=10, test_size=arg.validation)
            accuracy_3600, precision_3600, recall_3600, F1_3600, matrix_3600, percentage_matrix_3600, best_model_3600 = while_XGBoost_classification_tunned(df_features_3600, best_hyperparams_3600, cycles=10, test_size=arg.validation)
            accuracy_1800, precision_1800, recall_1800, F1_1800, matrix_1800, percentage_matrix_1800, best_model_1800 = while_XGBoost_classification_tunned(df_features_1800, best_hyperparams_1800, cycles=10, test_size=arg.validation)
            accuracy_900, precision_900, recall_900, F1_900, matrix_900, percentage_matrix_900, best_model_900 = while_XGBoost_classification_tunned(df_features_900, best_hyperparams_900, cycles=10, test_size=arg.validation)
            accuracy_600, precision_600, recall_600, F1_600, matrix_600, percentage_matrix_600, best_model_600 = while_XGBoost_classification_tunned(df_features_600, best_hyperparams_600, cycles=10, test_size=arg.validation)


            print("  validate best models")
            accuracy_validation_86400, precision_validation_86400, recall_validation_86400, F1_validation_86400,  matrix_validation_86400, percentage_matrix_validation_86400, feat_importances_86400 = validate_model(best_model_86400, df_validation_86400)
            accuracy_validation_43200, precision_validation_43200, recall_validation_43200, F1_validation_43200,  matrix_validation_43200, percentage_matrix_validation_43200, feat_importances_43200 = validate_model(best_model_43200, df_validation_43200)
            accuracy_validation_21600, precision_validation_21600, recall_validation_21600, F1_validation_21600,  matrix_validation_21600, percentage_matrix_validation_21600, feat_importances_21600 = validate_model(best_model_21600, df_validation_21600)
            accuracy_validation_14400, precision_validation_14400, recall_validation_14400, F1_validation_14400,  matrix_validation_14400, percentage_matrix_validation_14400, feat_importances_14400 = validate_model(best_model_14400, df_validation_14400)
            accuracy_validation_7200, precision_validation_7200, recall_validation_7200, F1_validation_7200,  matrix_validation_7200, percentage_matrix_validation_7200, feat_importances_7200 = validate_model(best_model_7200, df_validation_7200)
            accuracy_validation_3600, precision_validation_3600, recall_validation_3600, F1_validation_3600,  matrix_validation_3600, percentage_matrix_validation_3600, feat_importances_3600 = validate_model(best_model_3600, df_validation_3600)
            accuracy_validation_1800, precision_validation_1800, recall_validation_1800, F1_validation_1800,  matrix_validation_1800, percentage_matrix_validation_1800, feat_importances_1800 = validate_model(best_model_1800, df_validation_1800)
            accuracy_validation_900, precision_validation_900, recall_validation_900, F1_validation_900,  matrix_validation_900, percentage_matrix_validation_900, feat_importances_900 = validate_model(best_model_900, df_validation_900)
            accuracy_validation_600, precision_validation_600, recall_validation_600, F1_validation_600,  matrix_validation_600, percentage_matrix_validation_600, feat_importances_600 = validate_model(best_model_600, df_validation_600)
    
            print("  write")
            writer.writerows([
                [k, 86400, accuracy_86400, precision_86400, recall_86400, F1_86400, matrix_86400.tolist(), percentage_matrix_86400.tolist(), accuracy_validation_86400, precision_validation_86400, recall_validation_86400, F1_validation_86400,  matrix_validation_86400.tolist(), percentage_matrix_validation_86400.tolist(), feat_importances_86400],
                [k, 43200, accuracy_43200, precision_43200, recall_43200, F1_43200, matrix_43200.tolist(), percentage_matrix_43200.tolist(), accuracy_validation_43200, precision_validation_43200, recall_validation_43200, F1_validation_43200,  matrix_validation_43200.tolist(), percentage_matrix_validation_43200.tolist(), feat_importances_43200],
                [k, 21600, accuracy_21600, precision_21600, recall_21600, F1_21600, matrix_21600.tolist(), percentage_matrix_21600.tolist(), accuracy_validation_21600, precision_validation_21600, recall_validation_21600, F1_validation_21600,  matrix_validation_21600.tolist(), percentage_matrix_validation_21600.tolist(), feat_importances_21600],
                [k, 14400, accuracy_14400, precision_14400, recall_14400, F1_14400, matrix_14400.tolist(), percentage_matrix_14400.tolist(), accuracy_validation_14400, precision_validation_14400, recall_validation_14400, F1_validation_14400,  matrix_validation_14400.tolist(), percentage_matrix_validation_14400.tolist(), feat_importances_14400],
                [k, 7200, accuracy_7200, precision_7200, recall_7200, F1_7200, matrix_7200.tolist(), percentage_matrix_7200.tolist(), accuracy_validation_7200, precision_validation_7200, recall_validation_7200, F1_validation_7200,  matrix_validation_7200.tolist(), percentage_matrix_validation_7200.tolist(), feat_importances_7200],
                [k, 3600, accuracy_3600, precision_3600, recall_3600, F1_3600, matrix_3600.tolist(), percentage_matrix_3600.tolist(), accuracy_validation_3600, precision_validation_3600, recall_validation_3600, F1_validation_3600,  matrix_validation_3600.tolist(), percentage_matrix_validation_3600.tolist(), feat_importances_3600],
                [k, 1800, accuracy_1800, precision_1800, recall_1800, F1_1800, matrix_1800.tolist(), percentage_matrix_1800.tolist(), accuracy_validation_1800, precision_validation_1800, recall_validation_1800, F1_validation_1800,  matrix_validation_1800.tolist(), percentage_matrix_validation_1800.tolist(), feat_importances_1800],
                [k, 900, accuracy_900, precision_900, recall_900, F1_900, matrix_900.tolist(), percentage_matrix_900.tolist(), accuracy_validation_900, precision_validation_900, recall_validation_900, F1_validation_900,  matrix_validation_900.tolist(), percentage_matrix_validation_900.tolist(), feat_importances_900],
                [k, 600, accuracy_600, precision_600, recall_600, F1_600, matrix_600.tolist(), percentage_matrix_600.tolist(), accuracy_validation_600, precision_validation_600, recall_validation_600, F1_validation_600,  matrix_validation_600.tolist(), percentage_matrix_validation_600.tolist(), feat_importances_600],
            ])
    
    

if __name__ == "__main__":
    main()


