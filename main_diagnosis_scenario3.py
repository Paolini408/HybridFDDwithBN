from utils.diagnosis_statistical_analysis import *

if __name__ == '__main__':
    path_data = './data/preprocessed_data'
    path_data_list = glob.glob(os.path.join(path_data, '*.csv'))
    normal_operation = get_normal_operation(path_data_list)

    folder_path = './data/fault_isolated_data/scenario3'
    datetime_column = 'Datetime'

    month_week = 5  # Set from april to november
    start_day = 1
    end_day = 7

    month_day = 4  # Set from april to november
    single_day = 4  # Set from day 1 to day 7

    # 1) Cooling coil analysis
    plot_ccv_signal(normal_operation, folder_path, month_week, start_day, end_day, 3)
    plot_cc_kpi_day(folder_path, month_day, single_day, single_day, 3)

    # 2) Economizer analysis
    plot_oaf_vs_oat(normal_operation, folder_path, month_week, start_day, end_day, 3)
    plot_eco_kpi_day(folder_path, month_day, single_day, single_day, 3)

    # 3) SA_TEMP sensor bias analysis
    plot_sat_bias_kpi_day(folder_path, month_day, single_day, single_day, 3)
