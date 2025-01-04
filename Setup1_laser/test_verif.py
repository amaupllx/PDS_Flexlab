import pandas as pd
import matplotlib.pyplot as plt

for i in range(5,9):
    file_path_l = f'data_clean/finish_file_{i}.csv'
    data = pd.read_csv(file_path_l)

    max_force = data['Force'].max()
    print(f'The maximum force for the test {i} is: {max_force:.6f}N')
    plt.plot(data['delta'],data['Volume'])
    plt.xlabel('Delta (m)')
    plt.ylabel('Volume (m3)')
    plt.show()

    plt.plot(data['delta'],data['Force'])
    plt.xlabel('Delta (m)')
    plt.ylabel('Force (in N)')
    plt.show()