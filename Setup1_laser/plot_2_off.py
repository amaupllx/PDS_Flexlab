import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_laser_data(number: int = 1, plot = False):
    file_path = f'data/data_laser_{number}.csv'
    data = pd.read_csv(file_path)
    data.rename(columns={'Distance;': 'Distance'}, inplace=True)
    data['Distance'] = data['Distance'].str.replace(';', '', regex=False).astype(float)  # standardize the data
    data['Distance'] = pd.to_numeric(data['Distance'], errors='coerce')                  # set data to numeric

    # Sampling rate and time column
    sampling_rate = 250 if number < 4 else 1000  # in Hz
    time_interval = 1 / sampling_rate
    data['Time'] = data.index * time_interval  # add a column with time

    # Linear regression for the two time ranges
    def compute_linear_regression(data, time_start, time_end):
        mask = (data['Time'] >= time_start) & (data['Time'] <= time_end)
        x = data.loc[mask, 'Time']
        y = data.loc[mask, 'Distance']
        slope, intercept, _, _, _ = linregress(x, y)
        return slope, intercept

    slope1, intercept1 = compute_linear_regression(data, 10, 25)
    slope2, intercept2 = compute_linear_regression(data, 60, 100)

    # Generate the regression lines
    time_range1 = np.linspace(10, 60, 500)
    line1 = slope1 * time_range1 + intercept1

    time_range2 = np.linspace(20, 100, 500)
    line2 = slope2 * time_range2 + intercept2

    # Find intersection point of the two lines
    intersection_time = (intercept2 - intercept1) / (slope1 - slope2)
    intersection_distance = slope1 * intersection_time + intercept1
    
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs = axs.flatten()
        # Plot the data and regression lines
        axs[0].plot(data['Time'], data['Distance'], label=f'Data Laser {number}')
        axs[0].plot(time_range1, line1, color='red', label="Regression Line 1")
        axs[0].plot(time_range2, line2, color='green', label="Regression Line 2")
        axs[0].scatter(intersection_time, intersection_distance, color='purple', zorder=5)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Distance (μm)')
        axs[0].set_title('Linear regression')
        axs[0].legend()
        axs[0].grid(True)

    data = data[data['Time'] >= intersection_time].copy()
    data['Time'] -= intersection_time
    data.reset_index(drop=True, inplace=True)
    data['Distance'] -= data['Distance'].loc[0]
    # Find the index where Distance < 250 for the first time
    first_below_threshold_idx = data[data['Distance'] < -250].index.min()
    data = data.loc[:first_below_threshold_idx].copy()

    window_size = 100
    data['Distance_MA'] = data['Distance'].rolling(window=window_size, center=True).mean() # distance of the top of the cylinder from initial position

    q = (2/3)*10**(-6)          #m3/s
    pi = 3.14                   #constant
    R_in = 42.3*10**(-3)
    R_out = 50.3*10**(-3)
    area_c = pi*(R_out**2 - R_in**2) 
    R_t = 7.00*10**(-2)
    area_t = pi*(R_t**2) - area_c   #area of water

    data['water_l'] = data['Time']*q/area_t # distance of the level of water compared initial position
    if plot:
        axs[1].plot(data['water_l']*1000, data['Distance'], color='grey', label=f'real values')
        axs[1].plot(data['water_l']*1000, data['Distance_MA'], color='blue', label=f'smoothed values')
        axs[1].set_xlabel('Delta Distance (mm)')
        axs[1].set_ylabel('H - H\' (μm)')
        axs[1].set_title('Adjusted scale')
        axs[1].grid(True)
        axs[1].legend()

    data['delta'] = data['water_l'] + data['Distance_MA']/1000000       # in m, distance between top of cylinder and level of water
    data = data.dropna(subset=['delta'])
    data.to_csv(f'data_clean/data_laser_{number}.csv')

    if plot:
        axs[2].plot(data['water_l'], data['delta'])
        axs[2].set_xlabel('Water Level Δ (m)')
        axs[2].set_ylabel('Cylinder-Top Δ (m)')
        axs[2].set_title('Delta between water level and distance between top ring and water level')
        axs[2].grid(True)

    h = 12*10**(-3)             #m
    F_NM = 0.284                #N/m
    rho_water = 997             #kg/m3
    rho_vps = 1170              #kg/m3
    rho_air = 1.204             #kg/m3
    g = 9.81                    #kg.m/s2

    force = pi*g*(R_out**2 - R_in**2)*h*(rho_vps-rho_water)
    print(f'Weight - Buoyancy is :{force:.3f}N')

    data['Force']= -pi*g*(R_out**2 - R_in**2)*((h-data['delta'])*(rho_water-rho_air)-h*rho_vps)+F_NM*2*pi*(R_out+R_in)
    max_force = data['Force'].max()
    print(f'The maximum force for the test {number} is: {max_force:.3f}N')
    if plot:
        axs[3].plot(data['delta'], data['Force'])
        axs[3].set_xlabel('Cylinder-Top Δ (m)')
        axs[3].set_ylabel('Force (N)')
        axs[3].set_title('Force in respect of Δ')
        axs[3].grid(True)
        plt.tight_layout()
        plt.show()

for i in range(5,9):
    plot_laser_data(i, plot = False)