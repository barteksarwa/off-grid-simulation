# src/simulation/core.py
import numpy as np
import os
import pandas as pd
from datetime import datetime
from src.utils.sim_constants import *
from src.devices.photovoltaic import PhotovoltaicPanel
from src.devices.soe import SolidOxideElectrolyser
from src.devices.sofc import SolidOxideFuelCell
from src.devices.bess import Battery
from src.simulation.ems import AdvancedEMS

# Constants
SOC_INITIAL_H2 = 1.0
SOC_INITIAL_BESS = 0.2
H2_tank_capacity = 1000  # Adjusted to match context
a_cell = 0.01
t = 1073
parameters = {'G_stc': 1000, 'T_stc': 25, 'P_nom': 300}

def load_weather_data(file_path):
    """Load and process weather data"""
    try:
        df_weather = pd.read_excel(file_path, index_col=0, usecols='A, C:E, T')
        df_weather['IRRAD'] = df_weather['IDIFF_H'] + df_weather['IBEAM_H']
        case = df_weather[["IRRAD", "TAMB"]].to_numpy()
        time = df_weather.index.to_numpy()
        load = df_weather['E_el_HH'].to_numpy() * 1000
        return case, time, load
    except Exception as e:
        print(f"Error loading weather data: {e}")
        return None, None, None

def f(n_pv, n_bat, n_cells, n_tank):
    case, time, load = load_weather_data(r'data/input/load.xlsx')
    if case is None:
        return None, None
    
    pvg = PhotovoltaicPanel(parameters)
    bess_bat = Battery()
    sofc_dev = SolidOxideFuelCell(n_cells)
    soec_dev = SolidOxideElectrolyser(n_cells)
    ems_controller = AdvancedEMS()
    
    capacityh2 = n_tank * H2_tank_capacity
    ub = 1280 * n_bat
    qb = ub
    
    print(f'Updated parameters: Battery energy rating: {ub}, BESS power rating {qb}, \n'
          f'{n_pv} PV modules, The hydrogen tanks can store {n_tank * H2_tank_capacity} litres of H2, \n'
          f'Solid oxide stack has {n_cells} cells')

    power = np.zeros(len(load))
    for idx, (irrad, tamb) in enumerate(case):
        power[idx] = 0 if irrad <= 0 else n_pv * pvg.photovoltaic_power([(irrad, tamb)])
    net_power = power - load
    
    array_length = len(power) + 1
    lion_capacity = np.ones(array_length) * SOC_INITIAL_BESS
    prodh2 = np.zeros(array_length)
    consh2 = np.zeros(array_length)
    soch2 = np.ones(array_length) * SOC_INITIAL_H2
    deficit_energy = np.zeros(array_length)
    loss_energy = np.zeros(array_length)
    state = np.zeros(len(power))
    sofc_power = np.zeros(len(power))
    soec_power = np.zeros(len(power))
    battery_power = np.zeros(len(power))

    p_max_fc, j0_fc = sofc_dev.plot_fuel_cell_characteristic()
    j0_fc = j0_fc * 0.7
    p_max_fc = p_max_fc * 0.99
    P_sofc_base = ems_controller.P_sofc_base  # Baseload from EMS

    for i in range(len(power)):
        s = ems_controller.decide(load[i], power[i], lion_capacity[i], qb, soch2[i], i, ub)
        state[i] = s

        if s == 1:  # BESS charging
            charge_capacity = -bess_bat.bess_charge_capacity(qb, bc, ub, lion_capacity[i], SOC_max, 1)
            charge_capacity = np.minimum(float(charge_capacity), float(net_power[i]))
            dsoc = charge_capacity / ub
            dsoch2 = 0
            deficit_energy[i] = 0
            loss_energy[i] = max(0, net_power[i] - charge_capacity)
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = -charge_capacity 
             
        elif s == 2:  # SOEC producing hydrogen
            soec_power_max = max(0, net_power[i])
            j = soec_dev.newton_method(soec_dev.w_soec_diff, t, j0_fc, 101325, soec_power_max)
            hydrogen_possible_production = soec_dev.hydrogen_production_rate(j) * 22.4 * 3600
            if np.isnan(hydrogen_possible_production) or hydrogen_possible_production <= 0:
                prodh2[i] = 0
                soec_power[i] = 0
            else:
                prodh2[i] = np.minimum(hydrogen_possible_production, max(0, capacityh2 - soch2[i] * capacityh2))
                soec_power[i] = soec_power_max * prodh2[i] / hydrogen_possible_production if hydrogen_possible_production > 0 else 0
            dsoc = 0
            dsoch2 = prodh2[i] / capacityh2
            deficit_energy[i] = 0
            loss_energy[i] = max(0, net_power[i] - soec_power[i])
            sofc_power[i] = 0
            battery_power[i] = 0
            
        elif s == 3:  # BESS discharging
            discharge_capacity = bess_bat.bess_discharge_capacity(qb, bd, ub, lion_capacity[i], SOC_min, 1)
            discharge_capacity = np.minimum(float(discharge_capacity), float(-net_power[i]))
            dsoc = -discharge_capacity / ub
            dsoch2 = 0
            deficit_energy[i] = max(0, -net_power[i] - discharge_capacity)
            loss_energy[i] = 0
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = discharge_capacity 
                   
        elif s == 4:  # SOFC producing electricity
            p_fc = max(P_sofc_base, min(p_max_fc * n_cells * a_cell, abs(net_power[i])))
            j = sofc_dev.newton_method(sofc_dev.w_sofc_diff, t, j0_fc, 101325, p_fc)
            hydrogen_required = sofc_dev.hydrogen_consumption_rate(j) * 22.4 * 3600
            if np.isnan(hydrogen_required) or hydrogen_required <= 0 or soch2[i] < 0.1:
                consh2[i] = 0
                sofc_power[i] = 0
            else:
                consh2[i] = np.minimum(hydrogen_required, max(0, soch2[i] * capacityh2))
                sofc_power[i] = p_fc * consh2[i] / hydrogen_required if hydrogen_required > 0 else 0
            dsoc = 0
            dsoch2 = -consh2[i] / capacityh2
            deficit_energy[i] = max(0, abs(net_power[i]) - sofc_power[i])
            loss_energy[i] = 0
            soec_power[i] = 0
            battery_power[i] = 0

        elif s == 5:  # SOFC + BESS discharging
            discharge_capacity = bess_bat.bess_discharge_capacity(qb, bd, ub, lion_capacity[i], SOC_min, 1)
            discharge_capacity = np.minimum(float(discharge_capacity), float(-net_power[i]))
            net = net_power[i] + discharge_capacity
            p_fc = max(P_sofc_base, min(p_max_fc * n_cells * a_cell, abs(net)))
            j = sofc_dev.newton_method(sofc_dev.w_sofc_diff, t, j0_fc, 101325, p_fc)
            hydrogen_required = sofc_dev.hydrogen_consumption_rate(j) * 22.4 * 3600
            if np.isnan(hydrogen_required) or hydrogen_required <= 0 or soch2[i] < 0.1:
                consh2[i] = 0
                sofc_power[i] = 0
            else:
                consh2[i] = np.minimum(hydrogen_required, max(0, soch2[i] * capacityh2))
                sofc_power[i] = p_fc * consh2[i] / hydrogen_required if hydrogen_required > 0 else 0
            dsoc = -discharge_capacity / ub
            dsoch2 = -consh2[i] / capacityh2
            deficit_energy[i] = max(0, abs(net_power[i]) - sofc_power[i] - discharge_capacity)
            loss_energy[i] = 0
            soec_power[i] = 0
            battery_power[i] = discharge_capacity

        elif s == 6:  # SOEC + BESS charging
            charge_capacity = -bess_bat.bess_charge_capacity(qb, bc, ub, lion_capacity[i], SOC_max, 1)
            soec_power_max = max(0, net_power[i] - charge_capacity)
            j = soec_dev.newton_method(soec_dev.w_soec_diff, t, j0_fc, 101325, soec_power_max)
            hydrogen_possible_production = soec_dev.hydrogen_production_rate(j) * 22.4 * 3600
            if np.isnan(hydrogen_possible_production) or hydrogen_possible_production <= 0:
                prodh2[i] = 0
                soec_power[i] = 0
            else:
                prodh2[i] = np.minimum(hydrogen_possible_production, max(0, capacityh2 - soch2[i] * capacityh2))
                soec_power[i] = soec_power_max * prodh2[i] / hydrogen_possible_production if hydrogen_possible_production > 0 else 0
            dsoc = charge_capacity / ub
            dsoch2 = prodh2[i] / capacityh2
            deficit_energy[i] = 0
            loss_energy[i] = max(0, net_power[i] - soec_power[i] - charge_capacity)
            sofc_power[i] = 0
            battery_power[i] = charge_capacity
            
        elif s == 7:  # Deficit
            deficit_energy[i] = max(0, load[i] - power[i])
            dsoc = 0
            dsoch2 = 0
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = 0
            
        elif s == 8:  # Curtailment
            deficit_energy[i] = 0
            loss_energy[i] = max(0, power[i] - load[i])
            dsoc = 0
            dsoch2 = 0
            sofc_power[i] = 0
            soec_power[i] = 0
            battery_power[i] = 0

        else:
            dsoc = 0
            dsoch2 = 0
            print(f"Warning: Invalid state {s} at index {i}")

        lion_capacity[i+1] = max(0, min(1, lion_capacity[i] + dsoc))
        soch2[i+1] = max(0, min(1, soch2[i] + dsoch2))

    time_csv = np.array(time, dtype='datetime64')
    result = np.column_stack((
        lion_capacity[:-1], 
        power, 
        sofc_power,
        soec_power, 
        battery_power,
        load, 
        soch2[:-1],
        state,
        net_power,
        deficit_energy[:-1],
        loss_energy[:-1]
    ))

    return time_csv, result

if __name__ == "__main__":
    today_date = datetime.now().strftime('%Y%m%d')
    output = os.path.join('data', 'output', 'simulation', f'testride_{today_date}')
    os.makedirs(output, exist_ok=True)
    
    designs = [
        [72, 2, 27, 100],
        [73, 10, 13, 101],
    ]
    
    for idx, design in enumerate(designs):
        n_pv, n_bat, n_cells, n_tank = design
        text_csv_filename = os.path.join(output, f'pv_plot_{idx}.csv')
        
        print(f"\nSimulating design {idx}: {design}")
        time_csv, result = f(n_pv, n_bat, n_cells, n_tank)
        
        if result is None:
            print(f"Simulation failed for design {idx}")
            continue
            
        cumulative_sum_columns = np.nansum(result[:, -2:], axis=0)
        print(f"Cumulative sum for design {idx}: energy_deficit={cumulative_sum_columns[0]:.2f}, energy_loss={cumulative_sum_columns[1]:.2f}")
        
        try:
            np.savetxt(text_csv_filename, result, 
                      header='li_ion_capacity PV_power SOFC_power SOEC_power battery_power load SoCH2 EMS_State net_power energy_deficit energy_loss',
                      fmt='%.6f', delimiter=' ')
            print(f"Results saved to {text_csv_filename}")
        except Exception as e:
            print(f"Error saving results for design {idx} to {text_csv_filename}: {e}")