import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import polyfit, poly1d
from sklearn.metrics import r2_score
from scipy.integrate import quad

pi = 3.14                   #constant
R_in = 42.3*10**(-3)
R_out = 50.3*10**(-3)
h = 12*10**(-3)             #m
F_NM = 0.284                #N/m
rho_water = 997             #kg/m3
rho_vps = 1170              #kg/m3
rho_air = 1.204             #kg/m3
g = 9.81                    #kg.m/s2

# Load the area layer data
file_path = 'area_layers.csv'
layer = pd.read_csv(file_path)
layer['Area_m2'] = layer['Area_cm2'] * 1e-4  # Convert cm2 to m2
layer['Height_m'] = layer['Height_mm'] * 1e-3  # Convert mm to m

def best_polynomial_fit(x, y, max_degree=10):
    best_degree = 1
    best_r2 = -np.inf
    best_model = None

    for degree in range(1, max_degree + 1):

        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)
        
        y_pred = polynomial(x)
        
        # Calculate R² score
        r2 = r2_score(y, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_degree = degree
            best_model = polynomial
    
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, best_model(x), color='orange', label=f'Polynomial Fit')
    plt.axvline(x=0, color='r', linestyle='--', label='Top of the ring\n x = 0mm')
    plt.axvline(x=0.012, color='r', linestyle='--', label='Bottom of the ring\n x = 12mm')
    plt.xlabel('Distance from the top (m)')
    plt.ylabel('Area (m²)')
    plt.title(f'Area of the ring per layer')
    plt.legend()
    plt.grid(True)
    plt.show()
    return best_model, best_degree, best_r2

x = layer['Height_m']
y = layer['Area_m2']
best_model, best_degree, best_r2 = best_polynomial_fit(x, y, max_degree=80)

def integrate_polynomial(polynomial, a, b):
    def poly_function(x):
        return polynomial(x)

    integral, _ = quad(poly_function, a, b)
    return integral

total_ring_volume = integrate_polynomial(best_model, 0, h)
print(f'the total ring volume is {total_ring_volume}m3')

def compute_volume_for_df(df, polynomial):
    volumes = []
    for index, row in df.iterrows():
        start = row['delta']
        volume = integrate_polynomial(polynomial, start, h)
        volumes.append(volume)
    df['Volume'] = volumes

    return df

for i in range(5,9):
    file_path_l = f'data_clean/data_laser_{i}.csv'
    data = pd.read_csv(file_path_l)
    data = compute_volume_for_df(data, best_model)

    data['Force']= -g*data['Volume']*(rho_water-rho_air) +F_NM*2*pi*(R_out+R_in) + g*total_ring_volume*rho_vps
    data.to_csv(f'data_clean/finish_file_{i}.csv')
    max_force = data['Force'].max()
    print(f'The maximum force for the test {i} is: {max_force:.6f}N')