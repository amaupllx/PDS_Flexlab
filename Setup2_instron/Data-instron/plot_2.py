import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks
import numpy as np

def process_laser_data(test: str = 'angle', window_size: int = 8):
    angles = [0, 60, 120, 180, 240, 300]
    speeds = [2, 1.5, 1, 0.75, 0.5, 0.25]
    R_out = 50.3*10**(-3)       #m
    R_in = 42.4*10**(-3)        #m
    rho_water = 997             #kg/m3
    rho_vps = 1170              #kg/m3
    rho_air = 1.204             #kg/m3
    g = 9.81                    #kg.m/s2
    pi = 3.14
    h = 12*10**(-3)             #m

    constant_force = pi*g*(R_out**2 - R_in**2)*h*(rho_vps-(rho_water-rho_air))
    labels = angles if test == 'angle' else speeds
    processed_data = []
    x_intercepts = {}

    for label in labels:
        file_path_1 = f'Moncoque_{test}_{label}_1.csv'
        file_path_2 = f'Moncoque_{test}_{label}_2.csv'

        data_1 = pd.read_csv(file_path_1, skiprows=[1])
        data_2 = pd.read_csv(file_path_2, skiprows=[1])
        
        data_1 = data_1[data_1['Displacement'] >= 0.1].reset_index(drop=True)
        data_2 = data_2[data_2['Displacement'] >= 0.1].reset_index(drop=True)

        data_1['Force_MA'] = data_1['Force'].rolling(window=window_size, center=True).mean()
        data_2['Force_MA'] = data_2['Force'].rolling(window=window_size, center=True).mean()

        for data in [data_1, data_2]:
            mask = (data['Displacement'] >= 0.1) & (data['Displacement'] <= 0.5)
            valid_data = data[mask].dropna(subset=['Displacement', 'Force_MA'])

            if not valid_data.empty and len(valid_data) > 1:
                slope, _, _, _, _ = linregress(valid_data['Displacement'], valid_data['Force_MA'])
                #print(f"Slope for label {label}: {slope}")
                data['Force_MA_Adjusted'] = data['Force_MA'] - slope * data['Displacement']
                data['Force'] = data['Force'] - slope * data['Displacement']
            else:
                data['Force_MA_Adjusted'] = data['Force_MA']

            force_at_0_53_label = np.interp(0.53, data['Displacement'], data['Force_MA_Adjusted'])
            data['Force_MA_Adjusted'] = data['Force_MA_Adjusted'] - force_at_0_53_label
            data['Force'] = data['Force'] - force_at_0_53_label

            mask_regression = (data['Force_MA_Adjusted'] >= 0.01) & (data['Force_MA_Adjusted'] <= 0.03)
            regression_data = data[mask_regression].dropna(subset=['Displacement', 'Force_MA_Adjusted'])

            if not regression_data.empty and len(regression_data) > 1:
                reg_slope, reg_intercept, _, _, _ = linregress(
                    regression_data['Displacement'], regression_data['Force_MA_Adjusted']
                )

                if reg_slope != 0:
                    x_intercept = -reg_intercept / reg_slope
                    x_intercepts[f"{test}_{label}"] = x_intercept
                else:
                    x_intercepts[f"{test}_{label}"] = np.nan
            else:
                x_intercepts[f"{test}_{label}"] = np.nan
            data['Displacement'] = data['Displacement'] - x_intercept
            data = data[data['Displacement'] >= 0].reset_index(drop=True)
        data_1['Force_MA_Adjusted'] = data_1['Force_MA_Adjusted'] + constant_force
        data_1['Force'] = data_1['Force'] + constant_force

        data_2['Force_MA_Adjusted'] = data_2['Force_MA_Adjusted'] + constant_force
        data_2['Force'] = data_2['Force'] + constant_force

        processed_data.append({
            'label': label,
            'data_1': data_1,
            'data_2': data_2
        })
    return processed_data

def plot_laser_data(processed_data, test: str = 'angle'):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    title_suffix = "Angle" if test == 'angle' else "Speed"

    for i, data in enumerate(processed_data):
        label = data['label']
        data_1 = data['data_1']
        data_2 = data['data_2']

        ax = axes[i]

        ax.plot(data_1['Displacement'], data_1['Force'], color='grey')
        ax.plot(data_2['Displacement'], data_2['Force'], color='grey')

        ax.plot(data_1['Displacement'], data_1['Force_MA_Adjusted'], color='blue', label=f'Set 1')
        ax.plot(data_2['Displacement'], data_2['Force_MA_Adjusted'], color='orange', label=f'Set 2')
        
        ax.set_xlabel('Displacement (in mm)')
        ax.set_ylabel('Force (in N)')
        ax.set_title(f'Test {title_suffix} {label}°' if test == 'angle' else f'Test {title_suffix} {label}mm/s')
        ax.set_xlim((0, 0.5))
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    fig_compare, ax_compare = plt.subplots(figsize=(10, 6))

    for data in processed_data:
        label = data['label']
        data_1 = data['data_1']

        ax_compare.plot(data_1['Displacement'], data_1['Force_MA_Adjusted'], label=f'{title_suffix} {label}°' if test == 'angle' else f'{title_suffix} {label}mm/s')

    ax_compare.set_xlabel('Displacement (in mm)')
    ax_compare.set_ylabel('Force (in N)')
    ax_compare.set_title(f'Comparison of {test}s')
    ax_compare.set_xlim((0, 0.5))
    ax_compare.legend()
    ax_compare.grid(True)

    plt.tight_layout()
    plt.show()

def find_maxima_with_peaks(processed_data, prominence=0.005, height=0.01):
    maxima_results = {}

    for data_set in processed_data:
        label = data_set['label']
        maxima_results[label] = {}

        for set_id, data in enumerate([data_set['data_1'], data_set['data_2']], start=1):
            peaks, properties = find_peaks(
                data['Force_MA_Adjusted'], 
                prominence=prominence, 
                height=height
            )

            # Extract first local maximum if peaks are found
            if len(peaks) > 0:
                first_peak_index = peaks[0]
                first_local_max = data.iloc[first_peak_index]['Force_MA_Adjusted']
            else:
                first_local_max = np.nan  # No peaks found

            # Find the global maximum of the adjusted force data
            global_max = data['Force_MA_Adjusted'].max()

            # Store results
            maxima_results[label][f"Set_{set_id}"] = {
                'Global_Max': global_max,
                'First_Local_Max': first_local_max
            }

    return maxima_results

# 'angle'
processed_data_angle = process_laser_data(test='angle')
plot_laser_data(processed_data_angle, test='angle')

# 'speed'
processed_data_speed = process_laser_data(test='speed')
plot_laser_data(processed_data_speed, test='speed')

'''
maxima_results_s = find_maxima_with_peaks(processed_data_speed, prominence=0.0007, height=0.01)
maxima_results_a = find_maxima_with_peaks(processed_data_angle, prominence=0.001, height=0.01)

for label, sets in maxima_results_s.items():
    print(f"Label: {label}")
    for set_id, results in sets.items():
        print(f"  {set_id}: Global Max = {results['Global_Max']:.4f}, First Local Max = {results['First_Local_Max']:.4f}")
'''