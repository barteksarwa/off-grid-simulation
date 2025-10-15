"""
This script runs a simulation of a hybrid energy system in a loop. First it reads a table of designs generated using
Design of Experiments (DOE), gets the result for each design, and saves the results to CSV files.
The 'time' data is saved only for the first iteration.
"""

import numpy as np
import os
from src.simulation.core import f as simulation
from src.doe_analysis.doe_designs import doe_generate
from opt_fun_to_excel_crf import calculate_opt_fun
import optimization_result_crf as optimization_result

# Define the optimization range for each device
factors = [8, 80, 2, 20, 8, 80, 20, 200]  # PV_low, PV_high, BESS_low, BESS_high, rSOC_low, rSOC_high, HSS_low, HSS_high
doe_designs_file_name = 'doe_new.xlsx'  # design file name to save the design matrix in xlsx file

# Set the optimization parameters
# MAX_LPSR = 0  # maximum time allowed for the loss of power supply
# PENALTY_COEFF = 1e6
# PENALTY_COEFF_SURPLUS = 0
OBJECTIVE_FUN = -1 # Choose -1 for F1 multiobjective function,  -4 for F2 total cost, -3 for F3 deficit and loss

# Create output directory for the csv files with simulation results for each design
output_directory = 'doe_output_csv_new_pressure'
os.makedirs(output_directory, exist_ok=True)

# Generate full factorial designs
doe_normalized = doe_generate(factors, doe_designs_file_name)[0]
doe_denormalized = doe_generate(factors, doe_designs_file_name)[1]
design_table = doe_denormalized.to_numpy()


for i, row in enumerate(design_table):
    print(f'Simulating design no. {i}: ', *row)
    time_csv, output = simulation(*row)

    # Create file names with iteration number
    text_csv_filename = os.path.join(output_directory, f'pv_plot_{i}.csv')
    time_csv_filename = os.path.join(output_directory, f'time.csv')
    time_csv_str = np.datetime_as_string(time_csv)

    # Save files
    np.savetxt(text_csv_filename, output, header='li_ion_capacity PV_power SOFC_power SOEC_power battery_power load '
                                                 'SoCH2 EMS_State net_power energy_deficit energy_loss')
    if i == 1:  # Save 'time' only for the first iteration
        np.savetxt(time_csv_filename, time_csv_str, fmt='%s')

df_functions = calculate_opt_fun(output_directory, doe_normalized, doe_denormalized)
print(df_functions[:1])

features_const, target_values, target_lspr, target_surplus, quadratic_combinations = optimization_result.features_prep(df_functions, OBJECTIVE_FUN)
model, model_lspr, model_surplus = optimization_result.fit_the_model(features_const, target_values, target_lspr, target_surplus)
print(target_values)
print(target_lspr)
print(target_surplus)


# Display summary statistics
# print(f"Summary Statistics for Quadratic Regression Model:\n\n{model.summary()}\n\n")

bounds = [(-1,1),(-1,1),(-1,1),(-1,1)]
result = optimization_result.find_result(model, model_lspr, model_surplus, bounds)
print(result.x)

ranges = [(factors[i], factors[i + 1]) for i in range(0, len(factors), 2)]

denormalized_values = optimization_result.denormalize(result.x, ranges)
print(denormalized_values)
