import numpy as np
import pandas as pd
from math import log
from graphviz import Source
from sklearn import tree
from lineartree import LinearTreeRegressor
from lineartree import LinearForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib


def update_labels(dataframe, label_column):
    def modify_label(row):
        if row[label_column] == 'Normal':
            if row['label_detection'] in ['Normal old', 'Normal new']:
                return 'Normal True'
            elif row['label_detection'] == 'Fault':
                return 'Normal False'
        return row[label_column]

    dataframe[label_column] = dataframe.apply(modify_label, axis=1)
    return dataframe


def get_normal_operation(path_df_list, label_column):
    all_dataframe = []
    for path_df in path_df_list:
        dataframe = pd.read_csv(path_df)
        dataframe = update_labels(dataframe, label_column)
        dataframe = dataframe[dataframe[label_column] != 'Fault']
        all_dataframe.append(dataframe)
    if all_dataframe:
        return pd.concat(all_dataframe, ignore_index=True)
    else:
        print('Error')


def random_forest(train, validation, features_input, feature_output, hyper_param, performance_results, file_name, n_scenario):
    """
    Train a Random Forest regression model, perform hyperparameter tuning using GridSearchCV,
    evaluate the model on the validation set, and save the best model to disk.

    Args:
        train (DataFrame): Training data.
        validation (DataFrame): Validation data.
        features_input (list): List of feature columns.
        feature_output (str): Target output column.
        hyper_param (dict): Dictionary of hyperparameters for GridSearchCV.
        performance_results (DataFrame): DataFrame to store performance metrics.
        file_name (str): File name to save the trained model.
        n_scenario (int): Scenario number for saving the model.

    Returns:
        model.pkl: best model of random forest regressor.
    """
    input_train = train[features_input]
    target_train = train[feature_output]
    input_validation = validation[features_input]
    target_validation = validation[feature_output]
    regressor = RandomForestRegressor()
    Grid_search = GridSearchCV(regressor, hyper_param, cv=5)
    Grid_search.fit(input_train, target_train)
    model = Grid_search.best_estimator_
    joblib.dump(model, f'./models/scenario{n_scenario}/{file_name}')
    pred = model.predict(input_validation)
    MAE = mean_absolute_error(target_validation, pred)
    MSE = mean_squared_error(target_validation, pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(target_validation, pred)
    print(f"\nBest hyperparameters for {feature_output}: {Grid_search.best_params_}")
    print(f"Mean Absolute Error on validation data: {MAE}")
    print(f"Mean Squared Error on validation data: {MSE}")
    print(f"Root Mean Squared Error on validation data: {RMSE}")
    print(f"R2 score on validation data: {R2}")

    new_results = pd.DataFrame({'Target Feature': [feature_output],
                                'MAE': [MAE],
                                'MSE': [MSE],
                                'RMSE': [RMSE],
                                'R2': [R2]})
    new_results = new_results.dropna(axis=1, how='all')
    performance_results = performance_results.dropna(axis=1, how='all')
    performance_results = pd.concat([performance_results, new_results], ignore_index=True)
    return model, performance_results


def calculate_deviation(dataframe, selected_target, name_model, input_features_list, n_scenario):
    """
    Calculate the deviations between the actual and predicted values of the specified target variable,
    using the specified model for predictions.

    Args:
        dataframe (DataFrame): The DataFrame containing the data to calculate deviations on.
        selected_target (str): The column of the DataFrame representing the target variable to calculate deviations for.
        name_model (str): The name of the saved model used to make predictions.
        input_features_list (list): A list of column names from the DataFrame representing the input features used for predictions.
        n_scenario (int): The scenario number for saving the model.

    Returns:
        DataFrame: The original DataFrame with columns added for predictions and deviations.
    """
    selected_model = joblib.load(f'./models/scenario{n_scenario}/{name_model}.pkl')
    input_features = dataframe[input_features_list]
    prediction = selected_model.predict(input_features)
    dataframe['pred_' + selected_target] = prediction
    dataframe['dev_' + selected_target] = dataframe[selected_target] - prediction  # Actual - Predicted
    return dataframe


def compute_virtual_evidence(dataframe, metrics, selected_target):
    """
    This function computes the probability of fault (p_fault) for a selected target feature.

    Parameters:
    - dataframe: A pandas DataFrame containing the data on which the p_fault will be calculated.
    - metrics: A pandas DataFrame containing the RMSE values for different target features.
    - selected_target: A string representing the name of the target feature for which p_fault will be computed.

    Returns:
    - The modified DataFrame with an additional column for p_fault of the selected target feature.
    """
    rmse = metrics[metrics['Target Feature'] == selected_target]['RMSE'].values[0]*1  # It could be increased
    theta = -(3 ** 2) / 2 / log(0.5)
    res = dataframe['dev_' + selected_target]
    p_fault = 1 - np.exp(-(res ** 2) / (2 * theta * rmse ** 2))  # Supposing mean_res=0
    dataframe['p_fault_' + selected_target] = p_fault
    return dataframe
