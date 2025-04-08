import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import seaborn as sns
import calendar
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize, curve_fit
from utils.change_unit import fahrenheit_to_celsius


def modify_normal_labels(label):
    if label in ['Normal new', 'Normal old']:
        return 'Normal'
    else:
        return label


def filter_by_month_and_day(dataframe, m, day_start, day_end):
    return dataframe[(dataframe['Datetime'].dt.month == m) &
                     (dataframe['Datetime'].dt.day >= day_start) &
                     (dataframe['Datetime'].dt.day <= day_end)]


def filter_by_prediction(dataframe):
    filtered_df = dataframe.loc[dataframe['label_isolation'] == dataframe['label_isolation_pred']]
    return filtered_df


def filter_by_labels(df, fault_name):
    filtered_df = df[df['label_isolation_pred'].isin(['Normal', fault_name])]
    return filtered_df


def separate_by_mode(dataframe, mode_column='Operational_mode', mode_eco='Economizing mode'):
    eco_df = dataframe[dataframe[mode_column] == mode_eco]
    mech_df = dataframe[dataframe[mode_column] != mode_eco]
    return eco_df, mech_df


def get_normal_operation(path_df_list):
    all_dataframe = []
    for path_df in path_df_list:
        dataframe = pd.read_csv(path_df)
        dataframe['OA_fraction'] = (dataframe['RA_TEMP'] - dataframe['MA_TEMP']) / (dataframe['RA_TEMP'] - dataframe['OA_TEMP'])
        file_name = os.path.splitext(os.path.basename(path_df))[0]
        if file_name == 'AHU_annual':
            dataframe = dataframe[dataframe['label_detection'] == 'Normal old']
        else:
            dataframe = dataframe[dataframe['label_detection'] == 'Normal new']
        all_dataframe.append(dataframe)
    if all_dataframe:
        return pd.concat(all_dataframe, ignore_index=True)
    else:
        print('Error')


def get_regression_line_ccv(dataframe, degree=2):
    dataframe = dataframe[dataframe['Operational_mode'] != 'Economizing mode'].copy()
    dataframe['dt_cooling_coil'] = dataframe['SA_TEMP'] - dataframe['MA_TEMP']

    X = np.vstack([dataframe['CHWC_VLV_DM'] ** i for i in range(1, degree + 1)]).T
    y = dataframe['dt_cooling_coil']

    # Funzione obiettivo: errore quadratico della regressione
    def objective(coeffs):
        y_pred = X @ coeffs
        return np.sum((y - y_pred) ** 2)

    # Vincolo: f(1) > -19  (massimo dT aria in cooling)
    def constraint(coeffs):
        return sum(c * (1 ** (i + 1)) for i, c in enumerate(coeffs)) + 19

    # Stima iniziale dei coefficienti (soluzione senza vincoli)
    coeffs_init, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Ottimizzazione con vincolo
    result = minimize(objective, coeffs_init, constraints={'type': 'ineq', 'fun': constraint})

    coeffs = result.x
    # print(coeffs)

    # Funzione polinomiale con i coefficienti trovati
    def poly_eq(x):
        return sum(c * x ** (i + 1) for i, c in enumerate(coeffs))

    x_range = np.linspace(-1, 2, 1000)
    y_pred = poly_eq(x_range)

    # Calcolo degli errori residui
    residuals = dataframe['dt_cooling_coil'] - poly_eq(dataframe['CHWC_VLV_DM'])
    std_dev = np.std(residuals, ddof=1)

    return x_range, y_pred, std_dev


def plot_ccv_signal(df_normal, folder_path, m, day_start, day_end, n_scenario):
    csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_normal = df_normal[df_normal['Operational_mode'] != 'Economizing mode']

    x_normal, y_pred_normal, std_dev = get_regression_line_ccv(df_normal, degree=2)

    fig, ax = plt.subplots(nrows=1, figsize=(9, 6))
    ax.set_facecolor('whitesmoke')
    ax.scatter(df_normal['CHWC_VLV_DM'], df_normal['SA_TEMP'] - df_normal['MA_TEMP'], color='grey', s=35, alpha=0.15)
    ax.plot(x_normal, y_pred_normal, color='green', linestyle='-', linewidth=3, alpha=0.9)
    fill = ax.fill_between(x_normal, y_pred_normal - 3.5 * std_dev, y_pred_normal + 3.5 * std_dev, color='green', alpha=0.15)

    colors = ['darkgreen',
              '#aec7e8', '#6baed6', '#3182bd', '#1a5276',
              '#fb9a99', '#df6a78', '#a50f15', '#e41a1c',
              '#9e9ac8', '#756bb1', '#54278f', '#2c115f']

    normal = None
    ccv_list = []
    for idx, file in enumerate(csvs):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        color = colors[idx % len(colors)]

        df = filter_by_month_and_day(df, m, day_start, day_end)
        df = filter_by_prediction(df)
        df = filter_by_labels(df, 'Fault CC')

        if not df.empty:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            df['label_diagnosis'] = df['label_diagnosis'].apply(modify_normal_labels)
            eco_df, mech_df = separate_by_mode(df)

            if file_name.startswith('coi'):
                ccv = ax.scatter(mech_df['CHWC_VLV_DM'], mech_df['SA_TEMP'] - mech_df['MA_TEMP'], color=color, s=50)
                ccv_list.append(ccv)

            if file_name == 'AHU_annual':
                normal = ax.scatter(mech_df['CHWC_VLV_DM'], mech_df['SA_TEMP'] - mech_df['MA_TEMP'], color=color, s=50)

            ax.set_ylabel('SA_TEMP - MA_TEMP [°C]', fontsize=15)
            ax.set_xlabel('CHWC_VLV_DM', fontsize=15)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-16, 3)
            ax.grid(linewidth=0.2)

    legenda = fig.legend([fill, normal] + ccv_list,
                         ['Acceptability range', 'Normal data','CCV stuck 10%', 'CCV stuck 25%', 'CCV stuck 50%', 'CCV stuck 75%'],
                         loc='upper center', ncol=3, fontsize=13)
    legenda.get_frame().set_facecolor('whitesmoke')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f'./figs/diagnosis_scenario{n_scenario}/dTair_vs_ccvsignal_{m}_{day_start}-{day_end}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()


def plot_cc_kpi_day(folder_path, m, day_start, day_end, n_scenario):
    csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    fig, axes = plt.subplots(nrows=2, figsize=(9, 8), sharex=True)
    axes[0].set_facecolor('whitesmoke')
    axes[1].set_facecolor('whitesmoke')

    colors = ['darkgreen',
              '#aec7e8', '#6baed6', '#3182bd', '#1a5276',
              '#fb9a99', '#df6a78', '#a50f15', '#e41a1c',
              '#9e9ac8', '#756bb1', '#54278f', '#2c115f']

    normal = None
    ccv_list = []
    for idx, file in enumerate(csvs):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        color = colors[idx % len(colors)]

        df = filter_by_month_and_day(df, m, day_start, day_end)
        df = filter_by_prediction(df)
        df = filter_by_labels(df, 'Fault CC')

        if not df.empty:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            df['label_diagnosis'] = df['label_diagnosis'].apply(modify_normal_labels)
            eco_df, mech_df = separate_by_mode(df)

            # Plot for Economizing mode
            if file_name.startswith('damper_stuck_010'):
                axes[0].scatter(eco_df['Datetime'], eco_df['dev_SA_TEMP'], color='darkgreen', s=0)
            elif file_name == 'AHU_annual':
                normal = axes[0].scatter(eco_df['Datetime'], eco_df['dev_SA_TEMP'], color=color, s=50)
            else:
                ccv = axes[0].scatter(eco_df['Datetime'], eco_df['dev_SA_TEMP'], color=color, s=50)
                ccv_list.append(ccv)
            axes[0].set_ylabel('SA_TEMP model residual [°C]', fontsize=15)
            axes[0].set_title(f'Economizing mode (mode 2)', fontsize=18)
            axes[0].grid(linewidth=0.2)
            # axes[0].set_ylim(-5, 1)

            # Plot for Mechanical cooling mode
            if file_name.startswith('damper_stuck_010'):
                axes[1].scatter(mech_df['Datetime'], mech_df['SA_TEMP'] - mech_df['SA_TEMPSPT'], color='darkgreen', s=0)
            else:
                axes[1].scatter(mech_df['Datetime'], mech_df['SA_TEMP'] - mech_df['SA_TEMPSPT'], color=color, s=50)
            axes[1].set_ylabel('SA_TEMP - SA_TEMPSPT [°C]', fontsize=15)
            axes[1].set_xlabel('Time [h]', fontsize=15)
            axes[1].set_title(f'Mechanical cooling modes (mode 3 and 4)', fontsize=18)
            axes[1].grid(linewidth=0.2)
            # axes[1].set_ylim(-4, 8)

    if m < 10:
        fill = axes[0].fill_between(pd.date_range(start=f'2018-0{m}-01', end=f'2018-0{m}-10'), 0 - 0.35, 0 + 0.35, color='green', alpha=0.1)
        axes[1].fill_between(pd.date_range(start=f'2018-0{m}-01', end=f'2018-0{m}-10'), 0 - 0.95, 0 + 0.95, color='green', alpha=0.1)

        axes[0].set_xlim(pd.Timestamp(f'2018-0{m}-0{day_start} 05:00'), pd.Timestamp(f'2018-0{m}-0{day_end} 22:00'))
    else:
        fill = axes[0].fill_between(pd.date_range(start=f'2018-{m}-01', end=f'2018-{m}-10'), 0 - 0.35, 0 + 0.35, color='green', alpha=0.1)
        axes[1].fill_between(pd.date_range(start=f'2018-{m}-01', end=f'2018-{m}-10'), 0 - 0.95, 0 + 0.95, color='green', alpha=0.1)

        axes[0].set_xlim(pd.Timestamp(f'2018-{m}-0{day_start} 05:00'), pd.Timestamp(f'2018-{m}-0{day_end} 22:00'))

    date_formatter = mdates.DateFormatter('%H:%M')
    hour_locator = mdates.HourLocator(interval=3)
    axes[1].xaxis.set_major_formatter(date_formatter)
    axes[1].xaxis.set_major_locator(hour_locator)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

    legenda = fig.legend([fill, normal] + ccv_list,
                         ['Acceptability range', 'Normal data',
                          'CCV stuck 10%', 'CCV stuck 25%', 'CCV stuck 50%', 'CCV stuck 75%'],
                         loc='upper center', ncol=3, fontsize=13)
    legenda.get_frame().set_facecolor('whitesmoke')
    plt.tight_layout(rect=[0, 0, 1, 0.915])
    plt.savefig(f'./figs/diagnosis_scenario{n_scenario}/daily_kpi_ccv_{m}_{day_start}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()


def get_regression_line_oaf(dataframe, degree=2):
    dataframe = dataframe[dataframe['Operational_mode'] == 'Economizing mode'].copy()
    coeffs = np.polyfit(dataframe['OA_TEMP'], dataframe['OA_fraction'], deg=degree)
    poly_eq = np.poly1d(coeffs)

    # Calcolo degli errori residui
    residuals = dataframe['OA_fraction'] - poly_eq(dataframe['OA_TEMP'])
    std_dev = np.std(residuals, ddof=1)  # ddof=1 per la deviazione standard campionaria
    return poly_eq, std_dev


def oa_fraction_ideal(poly_eq, x):
    if x <= 0 or x >= 15.6:
        return 0.015  # oaf with minimum oad position
    elif 15.6 - 2 <= x < 15.6:  # 2°C in cui oad signal è = 1 in economizing mode (mode 2)
        return 1  # oaf with maximum oad position
    elif 0 < x < 15.6 - 2:
        return poly_eq(x)
    else:
        return None


def plot_oaf_vs_oat(df_normal, folder_path, m, day_start, day_end, n_scenario):
    csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    fig, ax = plt.subplots(nrows=1, figsize=(9, 6))
    ax.set_facecolor('whitesmoke')

    ax.scatter(df_normal['OA_TEMP'], df_normal['OA_fraction'], color='grey', s=35, alpha=0.15)
    poly_eq, std_dev = get_regression_line_oaf(df_normal, degree=2)
    y_fit_2 = [oa_fraction_ideal(poly_eq, x) for x in np.linspace(-50, 50, 100000)]
    ax.plot(np.linspace(-50, 50, 100000), y_fit_2, color='green', linestyle='-', linewidth=3, alpha=0.9)
    y_fit_sup_2 = [oa_fraction_ideal(poly_eq, x) + 3.5 * std_dev for x in np.linspace(0.01, 15.59, 100)]
    y_fit_inf_2 = [oa_fraction_ideal(poly_eq, x) - 3.5 * std_dev for x in np.linspace(0.01, 15.59, 100)]
    fill = ax.fill_between(np.linspace(0, 15.6, 100), y_fit_inf_2, y_fit_sup_2, color='green', alpha=0.15)

    colors = ['darkgreen',
              '#aec7e8', '#6baed6', '#3182bd', '#1a5276',
              '#fb9a99', '#df6a78', '#e41a1c', '#a50f15',
              '#9e9ac8', '#756bb1', '#54278f', '#2c115f']

    normal = None
    oad_list = []
    for idx, file in enumerate(csvs):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        color = colors[idx % len(colors)]

        df = filter_by_month_and_day(df, m, day_start, day_end)
        df = filter_by_prediction(df)
        df = filter_by_labels(df, 'Fault ECO')

        if not df.empty:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            df['label_diagnosis'] = df['label_diagnosis'].apply(modify_normal_labels)

            if file_name.startswith('damper_stuck_010'):
                oad_10_df_fault = df.loc[df['label_diagnosis'] != 'Normal']
                oad_10 = ax.scatter(oad_10_df_fault['OA_TEMP'], oad_10_df_fault['OA_fraction'], color=color, marker='o', s=50, alpha=0.5)
                oad_list.append(oad_10)

            elif file_name == 'AHU_annual':
                normal = ax.scatter(df['OA_TEMP'], df['OA_fraction'], s=50, color=color, marker='o', alpha=0.5)
            else:
                oad = ax.scatter(df['OA_TEMP'], df['OA_fraction'], s=50, color=color, marker='o', alpha=0.5)
                oad_list.append(oad)

            ax.set_ylabel('Outdoor Air Fraction (OAF)', fontsize=15)
            ax.set_xlabel('OA_TEMP [°C]', fontsize=15)
            ax.set_xlim(-10, 35)
            ax.set_ylim(-0.05, 1.1)
            ax.grid(linewidth=0.2)

    legenda = fig.legend([fill, normal] + oad_list,
                         ['Acceptability range', 'Normal data', 'OAD stuck 10%', 'OAD stuck 25%', 'OAD stuck 75%',
                          'OAD stuck 100%'],
                         loc='upper center', ncol=3, fontsize=13)
    legenda.get_frame().set_facecolor('whitesmoke')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f'./figs/diagnosis_scenario{n_scenario}/oaf_vs_oat_{m}_{day_start}-{day_end}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()


def plot_eco_kpi_day(folder_path, m, day_start, day_end, n_scenario):
    csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    fig, axes = plt.subplots(nrows=2, figsize=(9, 8), sharex=True)
    axes[0].set_facecolor('whitesmoke')
    axes[1].set_facecolor('whitesmoke')

    colors = ['darkgreen',
              '#aec7e8', '#6baed6', '#3182bd', '#1a5276',
              '#fb9a99', '#df6a78', '#e41a1c', '#a50f15',
              '#9e9ac8', '#756bb1', '#54278f', '#2c115f']

    normal = None
    eco_list = []
    for idx, file in enumerate(csvs):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        color = colors[idx % len(colors)]

        df = filter_by_month_and_day(df, m, day_start, day_end)
        df = filter_by_prediction(df)
        df = filter_by_labels(df, 'Fault ECO')

        if not df.empty:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            df['label_diagnosis'] = df['label_diagnosis'].apply(modify_normal_labels)
            eco_df, mech_df = separate_by_mode(df)

            # Plot for Economizing mode
            if file_name == 'AHU_annual':
                normal = axes[0].scatter(eco_df['Datetime'], eco_df['MA_TEMP'] - 12.78, color=color, s=50)
            else:
                axes[0].scatter(eco_df['Datetime'], eco_df['MA_TEMP'] - 12.78, color=color, s=50)
            axes[0].set_ylabel('MA_TEMP - MA_TEMPSPT [°C]', fontsize=15)
            axes[0].set_title(f'Economizing mode (mode 2)', fontsize=18)
            axes[0].grid(linewidth=0.2)
            # axes[0].set_ylim(-14, 2)

            # Plot for Mechanical cooling mode
            if file_name.startswith('damper_stuck_010'):
                normal_df_damper_10 = mech_df[mech_df['label_diagnosis'] == 'Normal']
                fault_df_damper_10 = mech_df[mech_df['label_diagnosis'] != 'Normal']
                axes[1].scatter(normal_df_damper_10['Datetime'], normal_df_damper_10['dev_MA_TEMP'], color='darkgreen', s=50)
                eco10 = axes[1].scatter(fault_df_damper_10['Datetime'], fault_df_damper_10['dev_MA_TEMP'], color=color, s=50)
                eco_list.append(eco10)
            elif file_name == 'AHU_annual':
                axes[1].scatter(mech_df['Datetime'], mech_df['dev_MA_TEMP'], color=color, s=50)
            else:
                eco = axes[1].scatter(mech_df['Datetime'], mech_df['dev_MA_TEMP'], color=color, s=50)
                eco_list.append(eco)
            axes[1].set_ylabel('MA_TEMP model residual [°C]', fontsize=15)
            axes[1].set_xlabel('Time [h]', fontsize=15)
            axes[1].set_title(f'Mechanical cooling modes (mode 3 and 4)', fontsize=18)
            axes[1].grid(linewidth=0.2)
            # axes[1].set_ylim(-24, 16)

    if m < 10:
        fill = axes[0].fill_between(pd.date_range(start=f'2018-0{m}-01', end=f'2018-0{m}-10'), 0 - 0.95, 0 + 0.95, color='green', alpha=0.1)
        axes[1].fill_between(pd.date_range(start=f'2018-0{m}-01', end=f'2018-0{m}-10'), 0 - 0.4, 0 + 0.4, color='green', alpha=0.1)
        axes[0].set_xlim(pd.Timestamp(f'2018-0{m}-0{day_start} 05:00'), pd.Timestamp(f'2018-0{m}-0{day_end} 22:00'))
    else:
        fill = axes[0].fill_between(pd.date_range(start=f'2018-{m}-01', end=f'2018-{m}-10'), 0 - 0.95, 0 + 0.95, color='green', alpha=0.1)
        axes[1].fill_between(pd.date_range(start=f'2018-{m}-01', end=f'2018-{m}-10'), 0 - 0.4, 0 + 0.4, color='green', alpha=0.1)
        axes[0].set_xlim(pd.Timestamp(f'2018-{m}-0{day_start} 05:00'), pd.Timestamp(f'2018-{m}-0{day_end} 22:00'))

    date_formatter = mdates.DateFormatter('%H:%M')
    hour_locator = mdates.HourLocator(interval=3)
    axes[1].xaxis.set_major_formatter(date_formatter)
    axes[1].xaxis.set_major_locator(hour_locator)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

    legenda = fig.legend([fill, normal] + eco_list,
                         ['Acceptability range', 'Normal data',
                          'OAD stuck 10%', 'OAD stuck 25%', 'OAD stuck 75%', 'OAD stuck 100%'],
                         loc='upper center', ncol=3, fontsize=13)
    legenda.get_frame().set_facecolor('whitesmoke')
    plt.tight_layout(rect=[0, 0, 1, 0.915])
    plt.savefig(f'./figs/diagnosis_scenario{n_scenario}/daily_kpi_eco_{m}_{day_start}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()


def plot_sat_bias_kpi_day(folder_path, m, day_start, day_end, n_scenario):
    csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    fig, axes = plt.subplots(nrows=2, figsize=(9, 8), sharex=True)
    axes[0].set_facecolor('whitesmoke')
    axes[1].set_facecolor('whitesmoke')

    colors = ['darkgreen',
              '#aec7e8', '#6baed6', '#3182bd', '#1a5276',
              '#fb9a99', '#df6a78', '#a50f15', '#e41a1c',
              '#9e9ac8', '#756bb1', '#54278f', '#2c115f']

    normal = None
    bias_list = []
    for idx, file in enumerate(csvs):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        color = colors[idx % len(colors)]

        df = filter_by_month_and_day(df, m, day_start, day_end)
        df = filter_by_prediction(df)
        df = filter_by_labels(df, 'Fault SAT bias')

        if not df.empty:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            df['label_diagnosis'] = df['label_diagnosis'].apply(modify_normal_labels)
            eco_df, mech_df = separate_by_mode(df)

            # Plot for Economizing mode
            if file_name.startswith('damper_stuck_010'):
                axes[0].scatter(eco_df['Datetime'], eco_df['SA_TEMP'] - eco_df['MA_TEMP'], color='darkgreen', s=0)
            elif file_name == 'AHU_annual':
                normal = axes[0].scatter(eco_df['Datetime'], eco_df['SA_TEMP'] - eco_df['MA_TEMP'], color=color, s=50)
            else:
                bias = axes[0].scatter(eco_df['Datetime'], eco_df['SA_TEMP'] - eco_df['MA_TEMP'], color=color, s=50)
                bias_list.append(bias)
            axes[0].set_ylabel('SA_TEMP - MA_TEMP [°C]', fontsize=15)
            axes[0].set_title(f'Economizing mode (mode 2)', fontsize=18)
            axes[0].grid(linewidth=0.2)
            # axes[0].set_ylim(-5, 5)

            # Plot for Mechanical cooling mode
            if file_name.startswith('damper_stuck_010'):
                axes[1].scatter(mech_df['Datetime'], mech_df['dev_CHWC_VLV_DM'], color='darkgreen', s=0)
            else:
                axes[1].scatter(mech_df['Datetime'], mech_df['dev_CHWC_VLV_DM'], color=color, s=50)
            axes[1].set_ylabel('CHWC_VLV_DM model residual', fontsize=15)
            axes[1].set_xlabel('Time [h]', fontsize=15)
            axes[1].set_title(f'Mechanical cooling modes (mode 3 and 4)', fontsize=18)
            axes[1].grid(linewidth=0.2)
            # axes[1].set_ylim(-0.51, 0.51)

    if m < 10:
        fill = axes[0].fill_between(pd.date_range(start=f'2018-0{m}-01', end=f'2018-0{m}-10'), 0 - 0.95, 0 + 0.95, color='green', alpha=0.1)
        axes[1].fill_between(pd.date_range(start=f'2018-0{m}-01', end=f'2018-0{m}-10'), 0 - 0.06, 0 + 0.06, color='green', alpha=0.1)
        axes[0].set_xlim(pd.Timestamp(f'2018-0{m}-0{day_start} 05:00'), pd.Timestamp(f'2018-0{m}-0{day_end} 22:00'))
    else:
        fill = axes[0].fill_between(pd.date_range(start=f'2018-{m}-01', end=f'2018-{m}-10'), 0 - 0.95, 0 + 0.95, color='green', alpha=0.1)
        axes[1].fill_between(pd.date_range(start=f'2018-{m}-01', end=f'2018-{m}-10'), 0 - 0.06, 0 + 0.06, color='green', alpha=0.1)
        axes[0].set_xlim(pd.Timestamp(f'2018-{m}-0{day_start} 05:00'), pd.Timestamp(f'2018-{m}-0{day_end} 22:00'))

    date_formatter = mdates.DateFormatter('%H:%M')
    hour_locator = mdates.HourLocator(interval=3)
    axes[1].xaxis.set_major_formatter(date_formatter)
    axes[1].xaxis.set_major_locator(hour_locator)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

    legenda = fig.legend([fill, normal] + bias_list,
                         ['Acceptability range', 'Normal data',
                          'SAT bias -2°C', 'SAT bias -4°C', 'SAT bias +2°C', 'SAT bias +4°C'],
                         loc='upper center', ncol=3, fontsize=13)
    legenda.get_frame().set_facecolor('whitesmoke')
    plt.tight_layout(rect=[0, 0, 1, 0.915])
    plt.savefig(f'./figs/diagnosis_scenario{n_scenario}/daily_kpi_sat_bias_{m}_{day_start}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
