import numpy as np
from utils.change_unit import fahrenheit_to_celsius


def modify_labels(label):
    if label in ['Normal new', 'Normal old']:
        return 'Normal'
    else:
        return label


def get_label_pred(row_df):
    fault_probs = {
        'Fault CC': row_df['CC'],
        'Fault SAT bias': row_df['SAT_SENSOR'],
        'Fault ECO': row_df['ECO'],
        'Fault MAT bias': row_df['MAT_SENSOR'],
        'Fault Air duct': row_df['Air duct']
    }
    # return max(fault_probs, key=fault_probs.get)
    sorted_probs = sorted(fault_probs.items(), key=lambda x: x[1], reverse=True)

    # Extract the top probabilities
    top_label, top_prob = sorted_probs[0]

    return top_label


def apply_daily_override_detection(dataframe):
    daily_counts = dataframe['label_detection_pred'].value_counts(normalize=True)
    max_label = daily_counts.idxmax()
    max_percentage = daily_counts.max()
    if max_percentage > 0.8:  # Based on domain knowledge
        dataframe['label_detection_pred'] = max_label
    return dataframe


def low_mixed_T(dataframe):
    threshold_temp = fahrenheit_to_celsius(55) - 5
    dataframe['label_isolation_pred'] = np.where(dataframe['MA_TEMP'] < threshold_temp, 'Fault ECO', dataframe['label_isolation_pred'])
    return dataframe


def apply_daily_override_isolation(dataframe):
    if dataframe['Operational_mode'].nunique() == 1:  # Rule for only one operational mode
        daily_counts = dataframe['label_isolation_pred'].value_counts(normalize=True)
        max_label = daily_counts.idxmax()
        max_percentage = daily_counts.max()
        if max_percentage > 0.8:  # Based on domain knowledge
            dataframe['label_isolation_pred'] = max_label
    else:  # Rule for different operational modes
        modes = dataframe['Operational_mode'].unique()
        mode_labels = {}
        for mode in modes:
            mode_df = dataframe[dataframe['Operational_mode'] == mode]
            mode_counts = mode_df['label_isolation_pred'].value_counts(normalize=True)
            max_label = mode_counts.idxmax()
            max_percentage = mode_counts.max()
            if max_percentage > 0.6:  # Based on domain knowledge
                mode_labels[mode] = max_label

        if len(mode_labels) == len(modes):
            if len(set(mode_labels.values())) == 1:
                dataframe['label_isolation_pred'] = next(iter(mode_labels.values()))  # In case of agreement
            else:
                if 'Mechanical cooling mode' in mode_labels and 'Mechanical cooling + Economizing mode' in mode_labels and 'Economizing mode' in mode_labels:
                    if mode_labels['Mechanical cooling mode'] == 'Fault ECO' or mode_labels['Mechanical cooling + Economizing mode'] == 'Fault ECO' or mode_labels['Economizing mode'] == 'Fault ECO':
                        dataframe['label_isolation_pred'] = 'Fault ECO'
                    elif (mode_labels['Mechanical cooling mode'] == 'Fault CC' or mode_labels['Mechanical cooling + Economizing mode'] == 'Fault CC') and mode_labels['Economizing mode'] == 'Fault SAT bias':
                        dataframe['label_isolation_pred'] = 'Fault CC'
                    elif (mode_labels['Mechanical cooling mode'] == 'Fault MAT bias' or mode_labels['Mechanical cooling + Economizing mode'] == 'Fault MAT bias') and mode_labels['Economizing mode'] != 'Fault MAT bias':
                        dataframe['label_isolation_pred'] = mode_labels['Economizing mode']

                elif 'Mechanical cooling mode' in mode_labels and 'Economizing mode' in mode_labels:
                    if mode_labels['Mechanical cooling mode'] == 'Fault ECO' or mode_labels['Economizing mode'] == 'Fault ECO':
                        dataframe['label_isolation_pred'] = 'Fault ECO'
                    elif mode_labels['Mechanical cooling mode'] == 'Fault CC' and mode_labels['Economizing mode'] == 'Fault SAT bias':
                        dataframe['label_isolation_pred'] = 'Fault CC'
                    elif mode_labels['Mechanical cooling mode'] == 'Fault MAT bias' and mode_labels['Economizing mode'] != 'Fault MAT bias':
                        dataframe['label_isolation_pred'] = mode_labels['Economizing mode']

                elif 'Mechanical cooling + Economizing mode' in mode_labels and 'Economizing mode' in mode_labels:
                    if mode_labels['Mechanical cooling + Economizing mode'] == 'Fault ECO' or mode_labels['Economizing mode'] == 'Fault ECO':
                        dataframe['label_isolation_pred'] = 'Fault ECO'
                    elif mode_labels['Mechanical cooling + Economizing mode'] == 'Fault CC' and mode_labels['Economizing mode'] == 'Fault SAT bias':
                        dataframe['label_isolation_pred'] = 'Fault CC'
                    elif mode_labels['Mechanical cooling + Economizing mode'] == 'Fault MAT bias' and mode_labels['Economizing mode'] != 'Fault MAT bias':
                        dataframe['label_isolation_pred'] = mode_labels['Economizing mode']
    return dataframe
