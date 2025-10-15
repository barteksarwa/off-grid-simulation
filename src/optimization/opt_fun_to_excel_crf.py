import os
import numpy as np
import pandas as pd
from datetime import datetime


def calculate_opt_fun(csv_directory, df_normalized, df_denormalized):
    csv_files = [f for f in os.listdir(csv_directory) if f.startswith('pv_plot') and f.endswith('.csv')]
    n_files = len(csv_files)

    designs_normalised = df_normalized.to_numpy()
    denom_designs_df = df_denormalized

    if not csv_files:
        print("No CSV files found in the directory.")
        return None, []

    dfs = []
    sum_positive_indexes_list = []
    dropped_files = []

    for i in range(n_files):
        file_path = os.path.join(csv_directory, 'pv_plot_' + str(i) + '.csv')
        df = pd.read_csv(file_path, delimiter=' ')
        if df.iloc[-1].isna().any():
            dropped_files.append(f'pv_plot_{i}.csv')
            df = df[:-1]
        sum_positive_indexes = np.sum(np.where(df.iloc[:, -3] > 0, 1, 0))
        sum_positive_indexes_list.append(sum_positive_indexes)
        df_np = df.to_numpy()
        s_row = np.sum(df_np, axis=0)
        dfs.append(s_row)

    dfs = np.array(dfs)
    dfs = dfs[:, -3:-1] / 1000  # Energy deficit and loss in kWh

    energy_deficit = dfs[:, 0:1]
    energy_loss = dfs[:, 1:2]
    
    energy_content_hydrogen = 33.3  # kWh / kg
    efficiency = 1
    hydrogen_deficit_kg = energy_deficit / (energy_content_hydrogen * efficiency)
    hydrogen_loss_kg = energy_loss / (energy_content_hydrogen * efficiency)

    costsunit_i = np.array([75, 323, 270, 325])  # Cost per unit for PV, Battery, SOFC, TANK
    costs_r = np.array([0, 4, 8, 0])  # Replacement cost factors
    costs_install = np.array([1000, 200 * costs_r[1], 500 * costs_r[2], 200])
    costs_m = np.array([100, 100, 200, 0])
    device_lifetimes = np.array([30, 7, 5, 30])  # years
    discount_rate = 0.05
    crfs = (discount_rate * (1 + discount_rate) ** device_lifetimes) / \
           ((1 + discount_rate) ** device_lifetimes - 1)

    capex_per_device = denom_designs_df.iloc[:, :4].values * costsunit_i
    capex_annualized = capex_per_device * crfs
    total_capex_annualized = np.sum(capex_annualized + denom_designs_df.iloc[:, :4].values * (costs_r * costsunit_i) + costs_install, axis=1)[:, np.newaxis]

    total_costs = total_capex_annualized + np.sum(costs_m) * 30  # Sum costs_m and broadcast

    LPSR = np.array([value / 8760 for value in sum_positive_indexes_list]).reshape(-1, 1)

    dfs = np.concatenate((
        designs_normalised,
        energy_deficit,
        energy_loss,
        hydrogen_deficit_kg,
        hydrogen_loss_kg,
        total_costs,
        LPSR
    ), axis=1)
    header = ["PV", "Battery", "SOFC", "TANK", "energy_deficit",
              "energy_loss", "hydrogen_deficit_kg", "hydrogen_loss_kg", "total_costs", "LPSR"]

    df_opt = pd.DataFrame(dfs, columns=header)

    # Save df_opt to CSV with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M")  # Format: YYYYMMDD_HHMM
    output_csv = os.path.join(csv_directory, f"df_opt_{current_time}.csv")
    try:
        df_opt.to_csv(output_csv, index=False)
        print(f"Debug: Successfully saved df_opt to {output_csv}")
    except Exception as e:
        print(f"Error saving df_opt to CSV: {e}")

    return df_opt, dropped_files

def calculate_total_cost(system_size):
    costsunit_i = np.array([75, 323, 270, 325])  # Cost per unit for PV, Battery, SOFC, TANK
    costs_r = np.array([0, 4, 8, 0])  # Replacement cost factors
    costs_install = np.array([1000, 200 * costs_r[1], 500 * costs_r[2], 200])  # [1000, 800, 4000, 200]
    costs_m = np.array([100, 100, 200, 0])  # Maintenance costs
    device_lifetimes = np.array([30, 7, 5, 30])  # Years
    discount_rate = 0.05

    # Calculate Capital Recovery Factor (CRF)
    crfs = (discount_rate * (1 + discount_rate) ** device_lifetimes) / \
           ((1 + discount_rate) ** device_lifetimes - 1)

    # Convert system_size to numpy array
    system_size = np.array(system_size)

    # Calculate capex per device
    capex_per_device = system_size * costsunit_i

    # Calculate annualized capex
    capex_annualized = capex_per_device * crfs

    # Calculate replacement costs
    replacement_costs = system_size * (costs_r * costsunit_i)

    # Calculate total annualized capex
    total_capex_annualized = np.sum(capex_annualized + replacement_costs + costs_install)

    # Calculate total maintenance cost (over 30 years)
    total_maintenance = np.sum(costs_m) * 30

    # Calculate total annualized cost
    total_cost = total_capex_annualized + total_maintenance

    return total_cost
if __name__ == "__main__":
    # csv_directory = r"C:\Users\Lenovo\Documents\python_projects\thesis\results_20250720_105600"
    # normalized_file = r"C:\Users\Lenovo\Documents\python_projects\thesis\READ ME\normalized_designs.csv"
    # denormalized_file = r"C:\Users\Lenovo\Documents\python_projects\thesis\READ ME\designs_val.csv"

    # try:
    #     df_normalized = pd.read_excel(normalized_file)
    #     df_denormalized = pd.read_excel(denormalized_file)
    # except Exception as e:
    #     print(f"Error reading Excel files: {e}")
    #     df_normalized = None
    #     df_denormalized = None

    # if df_normalized is not None and df_denormalized is not None:
    #     if not os.path.exists(csv_directory):
    #         print(f"Error: Directory {csv_directory} does not exist.")
    #     else:
    #         print(f"Directory found: {csv_directory}")
    #         try:
    #             df_results = calculate_opt_fun(csv_directory, df_normalized, df_denormalized)
    #             if df_results is not None:
    #                 print("NaN in df_results:\n", df_results.isna().sum())
    #                 print(df_results.head())
    #             else:
    #                 print("Function failed. Check previous error messages.")
    #         except Exception as e:
    #             print(f"Error: {e}")
    # else:
    #     print("Could not load Excel files. Check file paths and try again.")

    
    # 🧮 Calculate total cost for a specific system
    system_size = [80,	12,	36,	172]  # PV, Battery, SOFC, TANK
    total_cost = calculate_total_cost(system_size)
    print(f"Total annualized cost for system {system_size}: ${total_cost:.2f}")