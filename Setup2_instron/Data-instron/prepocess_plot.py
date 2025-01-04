import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import find_peaks
import numpy as np

window_size = 8
R_out = 50.3*10**(-3)       #m
R_in = 42.4*10**(-3)        #m
rho_water = 997             #kg/m3
rho_vps = 1170              #kg/m3
rho_air = 1.204             #kg/m3
g = 9.81                    #kg.m/s2
pi = 3.14
h = 12*10**(-3)             # m
V = 15.43*10**(-6)          # m3

constant_force_approx = pi*g*(R_out**2 - R_in**2)*h*(rho_vps-(rho_water-rho_air))
constant_force = V*g*(rho_vps-(rho_water-rho_air))

print(f'Approximative constant force: {constant_force_approx:.3f}N')
print(f'Constant force: {constant_force:.3f}N')

file_name = 'Moncoque_angle_300_2.csv'
data = pd.read_csv(file_name, skiprows=[1])

# First preprocessing
data = data[data['Displacement'] >= 0.1].reset_index(drop=True)

data['Force_MA'] = data['Force'].rolling(window=window_size, center=True).mean()

# Remove the buoyancy
mask = (data['Displacement'] >= 0.1) & (data['Displacement'] <= 0.5)
valid_data = data[mask].dropna(subset=['Displacement', 'Force_MA'])

if not valid_data.empty and len(valid_data) > 1:
    slope, intercept, _, _, _ = linregress(valid_data['Displacement'], valid_data['Force_MA'])
    #print(f"Slope for label {label}: {slope}")
    regression_line_b = slope * data['Displacement'] + intercept
    data['Force_MA_Adjusted'] = data['Force_MA'] - slope * data['Displacement']
else:
    data['Force_MA_Adjusted'] = data['Force_MA']

force_at_0_53_label = np.interp(0.53, data['Displacement'], data['Force_MA_Adjusted'])
data['Force_MA_Adjusted'] = data['Force_MA_Adjusted'] - force_at_0_53_label

# Shift the graph on the x-axis to 0
mask_regression = (data['Force_MA_Adjusted'] >= 0.01) & (data['Force_MA_Adjusted'] <= 0.03)
regression_data = data[mask_regression].dropna(subset=['Displacement', 'Force_MA_Adjusted'])

if not regression_data.empty and len(regression_data) > 1:
    reg_slope, reg_intercept, _, _, _ = linregress(
    regression_data['Displacement'], regression_data['Force_MA_Adjusted']
    )
    regression_line = reg_slope*data['Displacement'] + reg_intercept

if reg_slope != 0:
    x_intercept = -reg_intercept / reg_slope

data_f = pd.DataFrame()
data['Displacement_f'] = data['Displacement'] - x_intercept
data_f = data[data['Displacement_f'] >= 0].reset_index(drop=True)

# Plot
plt.figure(figsize=(14, 8))
plt.suptitle('Preprocessing Steps')

# First plot: Force with buoyancy
plt.subplot(2, 2, 1)
plt.plot(data['Displacement'], data['Force_MA'], label='Force', color='blue')
if not np.isnan(regression_line_b).all():
    plt.plot(data['Displacement'], regression_line_b, label='Force due to buoyancy', color='red', linestyle='--')
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
plt.title('1. Effect of buoyancy in the total force')
plt.legend()
plt.grid(True)

# Second plot: Force adjusted with add of Weight-Buoyancy
plt.subplot(2, 2, 2)
plt.plot(data['Displacement'], data['Force_MA_Adjusted'], color='blue')
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
plt.title('2. Force without the effect of the buoyancy on machine')
plt.grid(True)

# Third plot: Force with linear regression
plt.subplot(2, 2, 3)
plt.plot(data['Displacement'], data['Force_MA_Adjusted'], label='Adjusted Force', color='blue')
if not np.isnan(regression_line).all():
    plt.plot(data['Displacement'], regression_line, color='orange', linestyle='--')
plt.scatter(x_intercept, 0, color='red', zorder=5, label=f'X-axis Intercept: {x_intercept:.2f}')
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
plt.title('3. Force with the linear regression of the linear part')
plt.ylim([-0.003,0.073])
plt.legend()
plt.grid(True)

# Fourth plot: Force adjusted and shifted
plt.subplot(2, 2, 4)
plt.plot(data_f['Displacement_f'], data_f['Force_MA_Adjusted']+constant_force, color='green')
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
plt.title('4. Adjusted Force and Displacement, an addition of weight-buoyancy of the ring')
plt.grid(True)

plt.tight_layout()
plt.show()