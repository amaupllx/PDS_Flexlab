import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

test = pd.read_csv('Surface_tension_test_2.csv')

test = test[test['Displacement'] >= 10]
final_force = test[test['Displacement']>=53]['Force'].mean()

force_avg = test['Force'].groupby(test.index // 100).mean()
displacement_avg = test['Displacement'].groupby(test.index // 100).mean()

max_force_index = test['Force'].idxmax()

max_force = test.loc[max_force_index, 'Force']
max_displacement = test.loc[max_force_index, 'Displacement']

print(f"Maximum Force: {max_force} N")
print(f"Displacement of Maximum Force: {max_displacement} mm")
print(f"Final Force: {final_force:.3f} N")

surface_tension = max_force - final_force
print(f"Surface tension: {surface_tension:.3f} N")

surface_tension_m = surface_tension/(3.14*0.021)
print(f"Surface tension per meter: {surface_tension_m:.3f} N/m")

plt.figure(figsize=(8, 6))
plt.plot(displacement_avg, force_avg, linestyle='-', color='b', label='Average Force vs Displacement')
plt.title('Average Force vs Displacement', fontsize=16)
plt.xlabel('Displacement (mm)', fontsize=14)
plt.ylabel('Force (N)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()