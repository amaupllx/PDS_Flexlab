import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

def process_laser_data(test: str='angle', window_size: int=10):
    angles = [0, 60, 120, 180, 240, 300]
    speeds = [2, 1.5, 1, 0.75, 0.5, 0.25]
    
    labels = angles if test == 'angle' else speeds
    time_column = 'Time'
    delta = 0.02*window_size
    time_values = np.arange(0, 80, delta)
    data_frames = pd.DataFrame({time_column: time_values})
    
    for label in labels:
        file_name_1 = f'Moncoque_{test}_{label}_1.csv'
        file_name_2 = f'Moncoque_{test}_{label}_2.csv'

        data_1 = pd.read_csv(file_name_1, skiprows=[1])
        data_2 = pd.read_csv(file_name_2, skiprows=[1])

        data_1['Force_MA'] = data_1['Force'].rolling(window=window_size, center=True).mean()
        data_2['Force_MA'] = data_2['Force'].rolling(window=window_size, center=True).mean()

        data_1 = data_1[data_1['Displacement'] >= 0.1].reset_index(drop=True)
        data_2 = data_2[data_2['Displacement'] >= 0.1].reset_index(drop=True)

        data_frames[f'{label}_1'] = data_1['Force_MA']
        data_frames[f'disp_{label}_1'] = data_1['Displacement']
        data_frames[f'{label}_2'] = data_2['Force_MA']
        data_frames[f'disp_{label}_2'] = data_2['Displacement']

    return data_frames

def plot_laser_data(processed_df, test: str='angle'):
    labels = [0, 60, 120, 180, 240, 300] if test == 'angle' else [2, 1.5, 1, 0.75, 0.5, 0.25]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    title_suffix = "Angle" if test == 'angle' else "Speed"

    for i, label in enumerate(labels):
        ax = axes[i]

        # Extract data for plotting
        displacement_1 = processed_df[f'disp_{label}_1']
        force_ma_1 = processed_df[f'{label}_1']
        displacement_2 = processed_df[f'disp_{label}_2']
        force_ma_2 = processed_df[f'{label}_2']

        # Plot data
        ax.plot(displacement_1, force_ma_1, color='blue', label=f'Data (Set 1)')
        ax.plot(displacement_2, force_ma_2, color='orange', label=f'Data (Set 2)')
        
        ax.set_xlabel('Displacement (in mm)')
        ax.set_ylabel('Force (in N)')
        ax.set_title(f'Test {test} - {title_suffix} {label}°' if test == 'angle' else f'Test {test} - {title_suffix} {label}')
        ax.set_xlim((0.1, 1.2))
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Comparison plot
    fig_compare, ax_compare = plt.subplots(figsize=(10, 6))

    for label in labels:
        displacement = processed_df[f'disp_{label}_1']
        force_ma = processed_df[f'{label}_1']

        ax_compare.plot(displacement, force_ma, label=f'{title_suffix} {label}')

    ax_compare.set_xlabel('Displacement (in mm)')
    ax_compare.set_ylabel('Force (in N)')
    ax_compare.set_title(f'Comparison of {test}')
    ax_compare.set_xlim((0.1, 1.2))
    ax_compare.legend()
    ax_compare.grid(True)

    plt.tight_layout()
    plt.show()


combined_angle_df = process_laser_data(test='angle')
combined_speed_df = process_laser_data(test='speed')

combined_angle_df.to_csv('test')
