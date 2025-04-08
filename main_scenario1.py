import numpy as np
import pandas as pd
import os
import glob
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from utils.normal_identification import load_data, separate_by_mode, create_carpet_matrix, plot_carpet_matrix
from utils.baseline_establishment import update_labels, get_normal_operation, random_forest, calculate_deviation, compute_virtual_evidence
from utils.bn_structure_and_probs import obtain_models_mech, obtain_models_eco, obtain_evidence_mech, obtain_evidence_eco
from utils.postprocessing_prediction import modify_labels, get_label_pred, low_mixed_T, apply_daily_override_isolation

### SCENARIO 1: APAR RULES (only) ###
if __name__ == "__main__":
    n_scenario = 1
    # 1) Normal Identification
    folder_path_carpet = './data/preprocessed_data'
    datetime_column = 'Datetime'
    num_days = 365
    xticks = np.linspace(0, num_days * 24, 13)[:-1]
    xtick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colormap = ListedColormap(['green', 'red'])

    dataframes = load_data(folder_path_carpet, datetime_column)
    eco_dfs, mech_dfs = separate_by_mode(dataframes)

    file_names = ['Normal',
                  'CCV stuck 10%', 'CCV stuck 25%', 'CCV stuck 50%', 'CCV stuck 75%',
                  'OAD stuck 10%', 'OAD stuck 25%', 'OAD stuck 75%', 'OAD stuck 100%',
                  'SAT bias -2°C', 'SAT bias -4°C', 'SAT bias +2°C', 'SAT bias +4°C']
    colors_det = {'Normal': 0, 'Fault': 1}
    labels_det = ['Normal', 'Fault']
    label_column_det = 'label_detection_apar'  # Scenario 1
    carpet_matrix = create_carpet_matrix(dataframes, colors_det, label_column_det, num_days)
    plot_carpet_matrix(carpet_matrix, file_names, xticks, xtick_labels, labels_det, f'scenario{n_scenario}', colormap)

    # 2) Baseline Establishment
    performance_results = pd.DataFrame(columns=['Target Feature', 'MAE', 'MSE', 'RMSE', 'R2'])
    path_data = './data/training_and_validation_data'
    path_data_list = glob.glob(os.path.join(path_data, '*.csv'))
    df = get_normal_operation(path_data_list, label_column_det)

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['OA_fraction'] = (df['RA_TEMP'] - df['MA_TEMP']) / (df['RA_TEMP'] - df['OA_TEMP'])
    df['deltaT'] = df['SA_TEMPSPT'] - df['MA_TEMP']
    hyper = {'n_estimators': [100], 'max_depth': [6], 'random_state': [42]}

    # Splitting dataset in training and validation sets
    df_train, df_test = train_test_split(df, train_size=0.8, stratify=df[label_column_det], random_state=42)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # MA_TEMP model
    T_MA = 'MA_TEMP'
    features_input_T_MA = ['OA_TEMP', 'OA_DMPR_DM', 'RA_TEMP']
    best_model_T_MA, performance_results = random_forest(df_train, df_test, features_input_T_MA, T_MA, hyper,
                                                         performance_results, 'MA_TEMP.pkl', n_scenario)

    # SA_TEMP model
    T_SA = 'SA_TEMP'
    features_input_T_SA = ['OA_TEMP', 'RA_TEMP', 'MA_TEMP', 'CHWC_VLV_DM']
    best_model_T_SA, performance_results = random_forest(df_train, df_test, features_input_T_SA, T_SA, hyper,
                                                         performance_results, 'SA_TEMP.pkl', n_scenario)

    # CHWC_VLV_DM model
    CC_signal = 'CHWC_VLV_DM'
    features_input_CC_signal = ['OA_TEMP', 'RA_TEMP', 'MA_TEMP', 'SA_TEMP']
    best_model_CC_signal, performance_results = random_forest(df_train, df_test, features_input_CC_signal, CC_signal, hyper,
                                                              performance_results, 'CHWC_VLV_DM.pkl', n_scenario)

    # OA_DMPR_DM model
    OA_damper_signal = 'OA_DMPR_DM'
    features_input_OA_damper_signal = ['OA_TEMP', 'RA_TEMP', 'MA_TEMP']
    best_model_OA_damper_signal, performance_results = random_forest(df_train, df_test, features_input_OA_damper_signal, OA_damper_signal, hyper,
                                                                     performance_results, 'OA_DMPR_DM.pkl', n_scenario)

    # SF_WAT model
    SF_power = 'SF_WAT'
    features_input_SF_power = ['OA_TEMP', 'RA_TEMP', 'MA_TEMP', 'SA_TEMP', 'OA_DMPR_DM', 'CHWC_VLV_DM', 'SF_CS']
    best_model_SF_power, performance_results = random_forest(df_train, df_test, features_input_SF_power, SF_power, hyper,
                                                             performance_results, 'SF_WAT.pkl', n_scenario)

    # RF_WAT model
    RF_power = 'RF_WAT'
    features_input_RF_power = ['OA_TEMP', 'RA_TEMP', 'MA_TEMP', 'SA_TEMP', 'OA_DMPR_DM', 'CHWC_VLV_DM', 'RF_CS']
    best_model_RF_power, performance_results = random_forest(df_train, df_test, features_input_RF_power, RF_power, hyper,
                                                             performance_results, 'RF_WAT.pkl', n_scenario)

    # Save baseline performance metrics in csv file
    performance_results.to_csv(f'./results/baseline_models/scenario{n_scenario}.csv', index=False)

    # Calculate residuals + virtual evidence and then export the dataframes
    input_path = './data/testing_data/original'
    input_path_list = glob.glob(os.path.join(input_path, '*.csv'))
    output_path = f'./data/testing_data/scenario{n_scenario}'

    for file_path in input_path_list:
        df = pd.read_csv(file_path)
        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['OA_fraction'] = (df['RA_TEMP'] - df['MA_TEMP']) / (df['RA_TEMP'] - df['OA_TEMP'])
        df['deltaT'] = df['SA_TEMPSPT'] - df['MA_TEMP']

        calculate_deviation(df, 'MA_TEMP', 'MA_TEMP', features_input_T_MA, n_scenario)
        calculate_deviation(df, 'SA_TEMP', 'SA_TEMP', features_input_T_SA, n_scenario)
        calculate_deviation(df, 'CHWC_VLV_DM', 'CHWC_VLV_DM', features_input_CC_signal, n_scenario)
        calculate_deviation(df, 'OA_DMPR_DM', 'OA_DMPR_DM', features_input_OA_damper_signal, n_scenario)
        calculate_deviation(df, 'SF_WAT', 'SF_WAT', features_input_SF_power, n_scenario)
        calculate_deviation(df, 'RF_WAT', 'RF_WAT', features_input_RF_power, n_scenario)

        compute_virtual_evidence(df, performance_results, 'MA_TEMP')
        compute_virtual_evidence(df, performance_results, 'SA_TEMP')
        compute_virtual_evidence(df, performance_results, 'CHWC_VLV_DM')
        compute_virtual_evidence(df, performance_results, 'OA_DMPR_DM')
        compute_virtual_evidence(df, performance_results, 'SF_WAT')
        compute_virtual_evidence(df, performance_results, 'RF_WAT')

        output_file = os.path.basename(file_path)
        output_file_path = os.path.join(output_path, output_file)
        df.to_csv(output_file_path, index=False)

    # 3) Definition of BN structure and probabilities. Bayesian Inference.
    sorted_labels_detection = ['Normal', 'Fault']
    sorted_labels_detection.sort()

    model_TOT_mech, model_CC_mech, model_ECO_mech, model_SENSOR_mech, model_MAT_mech, model_DUCT_mech = obtain_models_mech()
    model_TOT_eco, model_CC_eco, model_ECO_eco, model_SENSOR_eco, model_MAT_eco, model_DUCT_eco = obtain_models_eco()

    all_df = []

    path = f'./data/testing_data/scenario{n_scenario}'
    path_file_list = glob.glob(os.path.join(path, '*.csv'))
    for file_path in path_file_list:
        df = pd.read_csv(file_path)
        df['label_detection'] = df['label_detection'].apply(modify_labels)
        df['label_isolation'] = df['label_isolation'].apply(modify_labels)
        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

        name_list = ['MA_TEMP', 'SA_TEMP', 'OA_DMPR_DM', 'CHWC_VLV_DM', 'SF_WAT', 'RF_WAT']
        metrics = pd.read_csv(f'./results/baseline_models/scenario{n_scenario}.csv')
        rmse_list = metrics.loc[metrics['Target Feature'].isin(name_list)]
        rmse_list = rmse_list.set_index('Target Feature')
        rmse_list = rmse_list.loc[name_list, 'RMSE'].values

        post_prob_df = pd.DataFrame(columns=['AHU', 'CC', 'ECO', 'SAT_SENSOR', 'MAT_SENSOR', 'Air duct'])

        for index, row in df.iterrows():
            if row['Operational_mode'] == 'Mechanical cooling mode' or row['Operational_mode'] == 'Mechanical cooling + Economizing mode':
                model_TOT_mech = model_TOT_mech
                model_CC_mech = model_CC_mech
                model_ECO_mech = model_ECO_mech
                model_SENSOR_mech = model_SENSOR_mech
                model_MAT_mech = model_MAT_mech
                model_DUCT_mech = model_DUCT_mech

                inference_mech = VariableElimination(model_TOT_mech)
                inference_CC_mech = VariableElimination(model_CC_mech)
                inference_ECO_mech = VariableElimination(model_ECO_mech)
                inference_SENSOR_mech = VariableElimination(model_SENSOR_mech)
                inference_MAT_mech = VariableElimination(model_MAT_mech)
                inference_DUCT_mech = VariableElimination(model_DUCT_mech)

                (virtual_TOT_mech, virtual_CC_mech, virtual_ECO_mech, virtual_DUCT_mech,
                 hard_TOT_mech, hard_CC_mech, hard_SENSOR_mech, hard_MAT_mech, hard_SP_mech) = obtain_evidence_mech(row, name_list, rmse_list)

                # Obtain posterior probability for each component
                phi_query_mech = inference_mech.query(['AHU'], evidence=hard_TOT_mech,
                                                      virtual_evidence=virtual_TOT_mech)
                post_prob_df.at[index, 'AHU'] = phi_query_mech.values[0]
                phi_query_CC_mech = inference_CC_mech.query(['Cooling Coil'], evidence=hard_CC_mech,
                                                            virtual_evidence=virtual_CC_mech)
                post_prob_df.at[index, 'CC'] = phi_query_CC_mech.values[0]
                phi_query_ECO_mech = inference_ECO_mech.query(['Economizer'], virtual_evidence=virtual_ECO_mech)
                post_prob_df.at[index, 'ECO'] = phi_query_ECO_mech.values[0]
                phi_query_SENSOR_mech = inference_SENSOR_mech.query(['SAT_Sensor'], evidence=hard_SENSOR_mech)
                post_prob_df.at[index, 'SAT_SENSOR'] = phi_query_SENSOR_mech.values[0]
                phi_query_MAT_mech = inference_MAT_mech.query(['MAT_Sensor'], evidence=hard_MAT_mech)
                post_prob_df.at[index, 'MAT_SENSOR'] = phi_query_MAT_mech.values[0]
                phi_query_DUCT_mech = inference_DUCT_mech.query(['Air duct'], evidence=hard_SP_mech,
                                                                virtual_evidence=virtual_DUCT_mech)
                post_prob_df.at[index, 'Air duct'] = phi_query_DUCT_mech.values[0]

            else:  # Economizing mode
                model_TOT_eco = model_TOT_eco
                model_CC_eco = model_CC_eco
                model_ECO_eco = model_ECO_eco
                model_SENSOR_eco = model_SENSOR_eco
                model_MAT_eco = model_MAT_eco
                model_DUCT_eco = model_DUCT_eco

                inference_eco = VariableElimination(model_TOT_eco)
                inference_CC_eco = VariableElimination(model_CC_eco)
                inference_ECO_eco = VariableElimination(model_ECO_eco)
                inference_SENSOR_eco = VariableElimination(model_SENSOR_eco)
                inference_MAT_eco = VariableElimination(model_MAT_eco)
                inference_DUCT_eco = VariableElimination(model_DUCT_eco)

                (virtual_TOT_eco, virtual_CC_eco, virtual_ECO_eco, virtual_DUCT_eco,
                 hard_TOT_eco, hard_ECO_eco, hard_SENSOR_eco, hard_MAT_eco, hard_SP_eco) = obtain_evidence_eco(row, name_list, rmse_list)

                phi_query = inference_eco.query(['AHU'], evidence=hard_TOT_eco, virtual_evidence=virtual_TOT_eco)
                post_prob_df.at[index, 'AHU'] = phi_query.values[0]
                phi_query_CC_eco = inference_CC_eco.query(['Cooling Coil'], virtual_evidence=virtual_CC_eco)
                post_prob_df.at[index, 'CC'] = phi_query_CC_eco.values[0]
                phi_query_ECO_eco = inference_ECO_eco.query(['Economizer'], evidence=hard_ECO_eco,
                                                            virtual_evidence=virtual_ECO_eco)
                post_prob_df.at[index, 'ECO'] = phi_query_ECO_eco.values[0]
                phi_query_SENSOR_eco = inference_SENSOR_eco.query(['SAT_Sensor'], evidence=hard_SENSOR_eco)
                post_prob_df.at[index, 'SAT_SENSOR'] = phi_query_SENSOR_eco.values[0]
                phi_query_MAT_eco = inference_MAT_eco.query(['MAT_Sensor'], evidence=hard_MAT_eco)
                post_prob_df.at[index, 'MAT_SENSOR'] = phi_query_MAT_eco.values[0]
                phi_query_DUCT_eco = inference_DUCT_eco.query(['Air duct'], evidence=hard_SP_eco,
                                                              virtual_evidence=virtual_DUCT_eco)
                post_prob_df.at[index, 'Air duct'] = phi_query_DUCT_eco.values[0]

        df_tot = pd.concat([df, post_prob_df], axis=1)

        # 4) Post-processing the results for fault detection and isolation (daily level)
        df_tot['Date'] = df_tot['Datetime'].dt.date

        df_tot['label_detection_pred'] = np.where(df_tot['AHU'] >= 0.5, 'Fault', 'Normal')  # Rule for detection

        df_tot['label_isolation_pred'] = np.where(df_tot['label_detection_pred'] == 'Normal', 'Normal', df_tot.apply(get_label_pred, axis=1))  # Rule for isolation

        df_tot = low_mixed_T(df_tot)  # Override for very low mixed temperature only during winter months

        df_tot = df_tot.groupby('Date').apply(apply_daily_override_isolation).reset_index(drop=True)  # Daily isolation expert-rules

        all_df.append(df_tot)

    df_final = pd.concat(all_df, ignore_index=True)

    # Detection results
    accuracy = accuracy_score(df_final['label_detection'], df_final['label_detection_pred'])
    print(f"\nAccuracy for detection =", accuracy * 100, '%')

    conf_matrix = confusion_matrix(df_final['label_detection'], df_final['label_detection_pred'])  # True labels in rows and predicted labels in columns
    conf_matrix_df = pd.DataFrame(conf_matrix, index=sorted_labels_detection, columns=sorted_labels_detection)
    conf_matrix_df.to_csv(f'./results/fault_detection/scenario{n_scenario}.csv')
    print(f"\nConfusion Matrix fo detection:")
    print(conf_matrix_df)

    # Isolation results
    accuracy_isolation = accuracy_score(df_final['label_isolation'], df_final['label_isolation_pred'])
    print(f"\nAccuracy for isolation=", accuracy_isolation * 100, '%')

    sorted_labels_isolation_true = sorted(df_final['label_isolation'].unique())
    sorted_labels_isolation_pred = sorted(df_final['label_isolation_pred'].unique())
    sorted_labels_isolation = max(sorted_labels_isolation_true, sorted_labels_isolation_pred, key=len)

    conf_matrix_isolation = confusion_matrix(df_final['label_isolation'], df_final['label_isolation_pred'])
    conf_matrix_isolation_df = pd.DataFrame(conf_matrix_isolation, index=sorted_labels_isolation,
                                            columns=sorted_labels_isolation)
    conf_matrix_isolation_df.to_csv(f'./results/fault_isolation/scenario{n_scenario}.csv')
    print(f"\nConfusion Matrix for isolation:")
    print(conf_matrix_isolation_df)
