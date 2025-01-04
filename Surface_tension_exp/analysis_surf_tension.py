import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_surface_tension(test_number: str = '1'):
    file_name = f'Surface_tension_test_{test_number}.csv'
    test = pd.read_csv(file_name)
    
    # Filter the data where 'Displacement' is greater than or equal to 10
    test = test[test['Displacement'] >= 10]
    
    stop = 50
    if test_number == '2': stop = 53

    final_force = test[test['Displacement'] >= stop]['Force'].mean()
    
    # Group the data by every 10 values and calculate the average for both force and displacement
    force_avg = test['Force'].groupby(test.index // 100).mean()
    displacement_avg = test['Displacement'].groupby(test.index // 100).mean()
    
    # Find the maximum force and the corresponding displacement
    max_force_index = test['Force'].idxmax()
    max_force = test.loc[max_force_index, 'Force']
    max_displacement = test.loc[max_force_index, 'Displacement']
    
    # Print results
    #print(f"Maximum Force: {max_force:.3f} N")
    #print(f"Displacement of Maximum Force: {max_displacement:.3f} mm")
    #print(f"Final Force: {final_force:.3f} N")
    
    # Calculate surface tension
    surface_tension = max_force - final_force
    #print(f"Surface Tension: {surface_tension:.3f} N")
    
    # Surface tension per meter (assuming diameter of 0.021 meters as in the original example)
    surface_tension_m = surface_tension / (3.14 * 0.021)
    print(f"Surface Tension per meter: {surface_tension_m:.3f} N/m, for test {test_number}")
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(displacement_avg, force_avg, linestyle='-', color='b', label='Average Force vs Displacement')
    plt.title(f'Average Force vs Displacement of test {test_number}', fontsize=16)
    plt.xlabel('Displacement (mm)', fontsize=14)
    plt.ylabel('Force (N)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'test_{test_number}_force_vs_displacement.png', dpi=300, bbox_inches='tight')
    plt.show()


analyze_surface_tension('1')
analyze_surface_tension('2')