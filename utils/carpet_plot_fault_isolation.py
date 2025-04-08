import numpy as np
import matplotlib.pyplot as plt


def create_carpet_matrix_new(dfs, colors, label_column, num_hours=24):
    num_datasets = len(dfs)
    carpet_matrix = np.full((num_datasets, 7 * 12 * num_hours), np.nan)
    for idx, df in enumerate(dfs):
        for m in range(1, 12 + 1):
            filtered_df = df[df['Datetime'].dt.month == m]
            for _, row in filtered_df.iterrows():
                day = row['Datetime'].day - 1
                if day < 0 or day >= 7:
                    continue
                hour = row['Datetime'].hour
                time_index = (m - 1) * 7 * num_hours + day * num_hours + hour

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


def plot_single_carpet_matrix(carpet_matrix, df_names, xticks, xtick_labels, labels, colormap, figsize=(9, 5)):
    num_datasets = len(df_names)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[30, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[:, 1])
    ax1.set_facecolor('whitesmoke')

    # Carpet plot
    cax1 = ax1.imshow(carpet_matrix, aspect='auto', cmap=colormap, interpolation='nearest', vmin=0, vmax=1)
    ax1.set_yticks(range(num_datasets))
    ax1.set_yticklabels(df_names)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels, rotation=0, ha='center')

    for week in range(1, 13):
        ax1.axvline(week * 7 * 24 - 0.5, color='black', linewidth=0.75, linestyle='-')
    for df in range(0, num_datasets):
        ax1.axhline(df + 0.499, color='black', linewidth=0.75, linestyle='-')

    # Add colorbar
    cbar = fig.colorbar(cax1, cax=cax, ticks=np.linspace(0, 1, len(labels)), orientation='vertical')
    cbar.ax.set_yticklabels(labels)

    plt.tight_layout()
    if len(labels) == 4:
        plt.savefig('./figs/fault_isolation_scenario3/actual_ground_truth.png', dpi=300, bbox_inches='tight', transparent=False)
    else:
        plt.savefig('./figs/fault_isolation_scenario3/predicted_ground_truth.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
