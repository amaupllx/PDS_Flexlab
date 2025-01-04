import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_laser_data(number: int=1):
    file_path = f'data/data_laser_{number}.csv'
    data = pd.read_csv(file_path)
    data.rename(columns={'Distance;': 'Distance'}, inplace=True)
    data['Distance'] = data['Distance'].str.replace(';', '', regex=False).astype(float) # standardize the data
    data['Distance'] = pd.to_numeric(data['Distance'], errors='coerce')                 # set data to numeric
    #data['Distance'] = data['Distance'] - 22100
    sampling_rate = 250 if number < 4 else 1000                                         # in Hz
    time_interval = 1 / sampling_rate
    data['Time'] = data.index * time_interval                                           # add a column with time
    data['Delta_Distance'] = data['Distance'].diff().abs()
    max_delta_index = data['Delta_Distance'].idxmax()
    max_delta_value = data['Delta_Distance'].max()                                      # look for the drop of distance when the cylinder buckle
    
    start_index = max(max_delta_index - 50000, 0)
    end_index = max_delta_index

    data_plot = data.loc[start_index:end_index]                                         # focus around the collapse time
    #data_plot['Distance'] = data_plot['Distance']-data_plot['Distance'].iloc[0]
    #print(f"The index with the highest delta in Distance is: {max_delta_index}")
    #print(f"The highest delta in Distance is: {max_delta_value}")
    window_size = 100
    data_plot['Distance_MA'] = data_plot['Distance'].rolling(window=window_size, center=True).mean()
    
    plt.plot(data_plot['Time'], data_plot['Distance_MA'], label=f'Data Laser {number}')
    plt.scatter(data_plot.loc[max_delta_index, 'Time'], data_plot.loc[max_delta_index, 'Distance_MA'], color='red', label='Max Delta Point', zorder=5)
    plt.xlabel('Time (in s)')
    plt.ylabel('Distance (in micrometers)')
    plt.title(f'Laser Data test {number}')
    plt.legend()
    limit = 15
    #plt.ylim(-2*limit, limit)
    plt.grid(True)
    plt.show()

plot_laser_data(6)