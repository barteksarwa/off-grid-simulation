# Scientific constants
R = 8.314  # Universal gas constant [J/mol K]
F = 9.64853321233e4  # Faraday constant [C/mol]
avogadro_number = 6.022e23  # [mol^(-1)]
coulomb = 6.241509e18  # [electrons]
t = 1173 # [K]

# Thermodynamics properties of H2, H2O, O2
hydrogen_specific_entropy_820 = 18.19  # [J/(g·K)]
hydrogen_enthalpy_820 = 42.19  # [kJ/mol]
oxygen_specific_entropy_820 = 32.29  # [J/(g·K)]
oxygen_enthalpy_820 = 33.06  # [kJ/mol]
water_specific_entropy_820 = 37.54  # [J/(g·K)]
water_enthalpy_820 = -240.99  # [kJ/mol]
n_e = 2

# Solid oxide electrolyser electrochemical properties
g_a = 2.051e9  # [A/m^2] preexponential factor for anode
g_c = 1.344e10  # [A/m^2] preexponential factor for cathode
e_actc = 1e5  # [J/mol]
e_acta = 1.1e5  # [J/mol]
sigma_a = 5e-5  # [m]
sigma_c = 5e-5  # [m]
sigma_e = 20e-5  # [m]
a_a = 2.98e-5  # [-]
a_c = 8.11e-5  # [-]
a_e = 2.94e-5  # [-]
b_a = -1392  # [-]
b_c = 600  # [-]
b_e = 10350  # [-]
electrode_porosity = 0.48  # [-]
electrode_tortuosity = 5.4  # [-]
m_h2 = 0.002  # [g/mol] molar mass of h2
m_h2o = 0.018  # [g/mol] molar mass of h2o
m_o2 = 0.032  # [g/mol] molar mass of o2
m_n2 = 0.028  # [g/mol] molar mass of n2
m_h2o_h2 = 2/(1/m_h2+1/m_h2o)
m_n2_o2 = 2/(1/m_n2+1/m_o2)
x_h2 = 0.95  # [-] (molar fraction of Hydrogen inlet of FC Mode)
x_h2o = 1-x_h2  # [-] (molar fraction of Water inlet of FC Mode)
x_h2_ec = 0.05  # [-] (molar fraction of Hydrogen inlet of EC Mode)
x_h2o_ec = 1-x_h2_ec  # [-] (molar fraction of Water inlet of EC Mode)
x_o2 = 0.21  # [-] (molar fraction of oxygen inlet)
x_n2 = 0.79  # [-] (molar fraction of nitrogen inlet)
sigma_f_h2 = 6.12  # [-]
sigma_f_h2o = 13.1  # [-]
sigma_f_n2 = 18.5  # [-]
sigma_f_o2 = 16.3  # [-]
r_pore = 0.25e-6  # [m]
# r_pore from AlZahrani https://sci-hub.se/10.1016/j.ijhydene.2017.03.186

# RSOFC
a_cell = 0.01  # [m^2]

# Photovoltaic panel parameteres from the datasheet
# Datasheet https://wpmgreenenergy.com/wp-content/uploads/2019/08/SPP320M60B.pan_.pdf

parameters = {
    'Name': 'Jiangsu Sunport Power Corp. Ltd. SPP320M60B',
    'Date': '03/1/2024',
    'T_NOCT': 43,
    'N_s': 60,
    'I_sc_ref': 10.31,
    'V_oc_ref': 39.6,
    'I_mp_ref': 9.86,
    'V_mp_ref': 32.4,
    'alpha_sc': 0.0062,
    'beta_oc': -0.121,
    'a_ref': 1.428, # modified ideality factor for reference conditions based on De Soto
    'v_TSTC': 1.38e-23*(25+273.15)/1.602e-19,
    'I_L_ref': 10.36,
    'I_o_ref': 1e-10,
    'R_s': 0.27,
    'R_sh_ref': 570,
    'gamma_r': -0.36,
    'Version': 'MM106',
    'Technology': 'Mono-c-Si',
}

# Li-ion battery parameters

bd = 0.95  # [-] Discharge efficiency
bc = 0.95  # [-] Charge efficiency


# Energy storage system limitations / parameters
SOC_min = 0.2 # [-] Minimum allowable SoC BESS
SOC_max = 0.8 # [-] Maximum allowable SoC BESS
SOCH2_min = 0
H2_tank_capacity = 7000  # [dm^3], , based on https://www.h2planet.eu/nl/detail/myh2%C3%AF%C2%BF%C2%BD_7000_1


ub_default = 1000