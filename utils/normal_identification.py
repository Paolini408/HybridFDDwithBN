import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(folder_path, datetime_column='Datetime'):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, parse_dates=[datetime_column])
        df = df.loc[df['Operational_mode'] != 'Unoccupied mode']
        dataframes.append(df)
    return dataframes


def separate_by_mode(dataframes, mode_column='Operational_mode', mode_value='Economizing mode'):
    mode_dfs = []
    other_dfs = []
    for df in dataframes:
        mode_df = df[df[mode_column] == mode_value]
        other_df = df[df[mode_column] != mode_value]
        mode_dfs.append(mode_df)
        other_dfs.append(other_df)
    return mode_dfs, other_dfs


def create_carpet_matrix(dataframes, colors, label_column, num_days, num_hours=24):
    num_datasets = len(dataframes)
    carpet_matrix = np.full((num_datasets, num_days * num_hours), np.nan)
    for idx, df in enumerate(dataframes):
        for _, row in df.iterrows():
            day_of_year = row['Datetime'].dayofyear - 1
            hour = row['Datetime'].hour
            time_index = day_of_year * num_hours + hour

            label_value = row[label_column]
            if isinstance(label_value, str):
                if label_value in colors:
                    carpet_matrix[idx, time_index] = colors[label_value]
                else:
                    raise ValueError(f"Label not found: {label_value}")
            elif isinstance(label_value, (int, float)):
                carpet_matrix[idx, time_index] = label_value
            else:
                raise ValueError(f"Wrong data type {label_column}: {type(label_value)}")

    return carpet_matrix


def plot_carpet_matrix(carpet_matrix, file_names, xticks, xtick_labels, labels, info, colormap, figsize=(9, 5)):
    num_datasets = len(file_names)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_facecolor('whitesmoke')

    # Carpet plot
    cax = ax.imshow(carpet_matrix, aspect='auto', cmap=colormap, interpolation='nearest', vmin=0, vmax=1)
    ax.set_yticks(range(num_datasets))
    ax.set_yticklabels(file_names)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0)

    ax.axvline(8760/12, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*2, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*3, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*4, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*5, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*6, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*7, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*8, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*9, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*10, color='black', linewidth=0.75, linestyle='-')
    ax.axvline(8760/12*11, color='black', linewidth=0.75, linestyle='-')

    for df in range(0, num_datasets):
        ax.axhline(df + 0.499, color='black', linewidth=0.75, linestyle='-')

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, ticks=np.linspace(0, 1, len(labels)), orientation='vertical')
    cbar.ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig(f'./figs/normal_identification/{info}.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
