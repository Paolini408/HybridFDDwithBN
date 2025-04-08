import pandas as pd
import os
import glob


if __name__ == '__main__':
    folder_path = './data/preprocessed_data'
    path_data_list = glob.glob(os.path.join(folder_path, '*.csv'))

    output_path_testing = './data/testing_data/original'
    output_path_training = './data/training_and_validation_data'

    for file_path in path_data_list:
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

        # Get the week of the month
        df['Week'] = df['Datetime'].dt.isocalendar().week

        # Initialize list to hold indices of the first week of each month
        test_indices = []
        for month in range(1, 13):
            first_week_indices = df[(df['Month'] == month) & (df['Datetime'].dt.day <= 7)].index
            test_indices.extend(first_week_indices)

        testing_df = df.loc[test_indices]
        training_df = df.drop(test_indices)

        output_file = os.path.basename(file_path)

        # Testing
        output_file_path_testing = os.path.join(output_path_testing, output_file)
        testing_df.to_csv(output_file_path_testing, index=False)

        # Training
        output_file_path_training = os.path.join(output_path_training, output_file)
        training_df.to_csv(output_file_path_training, index=False)
