from pyDOE2 import fullfact
import pandas as pd


def doe_generate(factor_levels, file_name):
    # Unpack the factor levels
    pv_low, pv_high, bess_low, bess_high, rsoc_low, rsoc_high, HSS_low, HSS_high = factor_levels

    # Define the levels for each factor
    factor_levels = {
        "Photovoltaic Modules (number)": [i for i in range(pv_low, pv_high)],
        "Battery Power (number)": [i for i in range(bess_low, bess_high)],
        "Solid Oxide Stack (number of cells)": [i for i in range(rsoc_low, rsoc_high)],
        "Hydrogen Storage Tanks (number of tanks)": [i for i in range(HSS_low, HSS_high)]
    }

    # Create a list of factors and their corresponding levels
    factors = list(factor_levels.keys())

    # Generate a full factorial design
    design = fullfact([4,4,4,4])-1

    # Convert the design matrix to a DataFrame
    df = pd.DataFrame(design, columns=factors)

    # Print the design matrix
    df.to_excel(file_name, index=False)
    df_denormalized = df.copy()

    for factor in factors:
        lower_bound = min(factor_levels[factor])
        upper_bound = max(factor_levels[factor])
        scaling_factor = (upper_bound - lower_bound) / 2.0  # Scale factor is half of the range

        df_denormalized[factor] = df[factor].apply(lambda x: x * scaling_factor + (lower_bound + upper_bound) / 2)


    # Save the denormalized design matrix to an Excel file
    denormalized_excel_file_name = f'{file_name.split(".")[0]}_denormalized.xlsx'
    df_denormalized.to_excel(denormalized_excel_file_name, index=False)

    return df, df_denormalized