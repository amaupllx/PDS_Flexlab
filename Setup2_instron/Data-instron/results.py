import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Constant
pi = 3.1416         # valiue of Pi
E = 1.26e6          # in Pa
nu = 0.5            # constant
t = 0.231e-3  # in m
h = 12*10**(-3)             #m

R_out = 50.3*10**(-3)       #m
R_in = 42.4*10**(-3)        #m
rho_water = 997             #kg/m3
rho_vps = 1170              #kg/m3
rho_air = 1.204             #kg/m3
g = 9.81                    #kg.m/s2

constant_force = pi*g*(R_out**2 - R_in**2)*h*(rho_vps-(rho_water-rho_air))

F_theo = (2*pi*E*t**2)/math.sqrt(3*(1-nu**2))

print(f'The maximal theoritical force in {F_theo:.3f} N')

# Results from Instron test
data_speed = {
    "2": {"Local": [0.1049, 0.1068], "Total": [0.1232, 0.1204]},
    "1.5": {"Local": [0.1084, 0.0998], "Total": [0.1193, 0.1229]},
    "1": {"Local": [0.1017, 0.1008], "Total": [0.1202, 0.1201]},
    "0.75": {"Local": [0.0506, 0.1033], "Total": [0.1186, 0.1193]},
    "0.5": {"Local": [0.0971, 0.1020], "Total": [0.1169, 0.1170]},
    "0.25": {"Local": [0.0477, 0.0478], "Total": [0.1165, 0.1168]},
}

data_angle = {
    "0": {"Local": [0.1031, 0.0973], "Total": [0.1221, 0.1204]},
    "60": {"Local": [0.0866, 0.0879], "Total": [0.1188, 0.1203]},
    "120": {"Local": [0.0764, 0.0865], "Total": [0.1135, 0.1137]},
    "180": {"Local": [0.0818, 0.0800], "Total": [0.1112, 0.1114]},
    "240": {"Local": [0.0873, 0.0882], "Total": [0.1100, 0.1099]},
    "300": {"Local": [0.0963, 0.0938], "Total": [0.1152, 0.1166]},
}

df_F_speed_loc = pd.DataFrame({key: val["Local"] for key, val in data_speed.items()}).T
df_F_speed_loc.columns = ["Set_1", "Set_2"]

df_F_speed_tot = pd.DataFrame({key: val["Total"] for key, val in data_speed.items()}).T
df_F_speed_tot.columns = ["Set_1", "Set_2"]

df_F_angle_loc = pd.DataFrame({key: val["Local"] for key, val in data_angle.items()}).T
df_F_angle_loc.columns = ["Set_1", "Set_2"]

df_F_angle_tot = pd.DataFrame({key: val["Total"] for key, val in data_angle.items()}).T
df_F_angle_tot.columns = ["Set_1", "Set_2"]

F_speed_loc_mean = df_F_speed_loc.mean(axis=1)
F_angle_loc_mean = df_F_angle_loc.mean(axis=1)

K_speed_loc_mean = df_F_speed_loc.mean(axis=1)/F_theo
K_angle_loc_mean = df_F_angle_loc.mean(axis=1)/F_theo

y_min = 0.15
y_max = 0.40

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot F_angle_loc_mean
axes[0].plot(K_angle_loc_mean.index.astype(float), K_angle_loc_mean.values, marker='o', color='blue', label="Local Max (Angles)")
axes[0].set_title("For different Angles")
axes[0].set_xlabel("Angle (degrees)")
axes[0].set_ylabel("$\\kappa$")
axes[0].set_ylim((y_min,y_max))
axes[0].grid(True)
axes[0].legend()

ax0_twin = axes[0].twinx()
ax0_twin.plot(F_angle_loc_mean.index.astype(float), F_angle_loc_mean.values, marker='o', color='blue')
ax0_twin.set_ylabel("Force (in N)")
ax0_twin.set_ylim((y_min*F_theo,y_max*F_theo))

# Plot F_speed_loc_mean
axes[1].plot(K_speed_loc_mean.index.astype(float), K_speed_loc_mean.values, marker='o', color='green', label="Local Max (Speeds)")
axes[1].set_title("For different Speeds")
axes[1].set_xlabel("Speed (mm/min)")
axes[1].set_ylabel("$\\kappa$")
axes[1].set_ylim((y_min,y_max))
axes[1].grid(True)
axes[1].legend()

ax1_twin = axes[1].twinx()
ax1_twin.plot(F_speed_loc_mean.index.astype(float), F_speed_loc_mean.values, marker='o', color='green')
ax1_twin.set_ylabel("Force (in N)")
ax1_twin.set_ylim((y_min*F_theo,y_max*F_theo))

fig.suptitle('Evolution of $\\kappa$ and local maximal Force:')
plt.tight_layout()
plt.show()