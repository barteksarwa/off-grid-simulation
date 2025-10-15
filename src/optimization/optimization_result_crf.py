import os
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import differential_evolution
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


csv_directory = r"C:\Users\Lenovo\Documents\python_projects\thesis\project\doe_output_csv_5_levels2802"
normalized_file = r"C:\Users\Lenovo\Documents\python_projects\thesis\project\doe_new_5_levels.xlsx"
denormalized_file = r"C:\Users\Lenovo\Documents\python_projects\thesis\project\doe_new_5_levels_denormalized.xlsx"

try:
    df_normalized = pd.read_excel(normalized_file)
    df_denormalized = pd.read_excel(denormalized_file)
except Exception as e:
    print(f"Error reading Excel files: {e}")
    df_normalized = None
    df_denormalized = None


def features_prep(results, target_col=None):
    features = results.iloc[:, 0:4].values
    # Reduce quadratic terms for LPSR to improve modeling of LPSR=0
    poly = PolynomialFeatures(degree=7, include_bias=False) 
    features_poly = poly.fit_transform(features)
    target_values = results[target_col].values if target_col is not None and target_col in results.columns else None
    return features_poly, target_values, poly

def fit_the_model(features_poly, target_values):
    model = LinearRegression().fit(features_poly, target_values)
    r_squared = model.score(features_poly, target_values)
    print(f"Debug: Fitted Polynomial Regression model, R²={r_squared:.4f}")
    return model

def find_result(model_costs, model_lpsr, model_h2_cost, poly_costs, poly_lpsr, poly_h2_cost, bounds, weights, total_costs_min, total_costs_max, h2_cost_min, h2_cost_max):
    def f_opt(X):
        features = np.array(X)[np.newaxis, :]
        # Transform features for each model
        features_costs = poly_costs.transform(features)
        features_lpsr = poly_lpsr.transform(features)
        features_h2_cost = poly_h2_cost.transform(features)
        
        # Predictions
        total_costs = model_costs.predict(features_costs)
        lpsr = model_lpsr.predict(features_lpsr)
        h2_cost = model_h2_cost.predict(features_h2_cost)
        # Min-max normalization for total_costs
        total_costs_norm = (total_costs - total_costs_min) / (total_costs_max - total_costs_min)
        h2_cost_norm = (h2_cost - h2_cost_min) / (h2_cost_max - h2_cost_min) 
        objective = weights['total_costs'] * total_costs_norm + weights['lpsr'] * lpsr + weights['h2_cost'] * h2_cost_norm
        
        return objective
    
    result = differential_evolution(f_opt, bounds)
    return result


def denormalize_value(normalized_value, original_min, original_max):
    return original_min + (original_max - original_min) * (normalized_value + 1) / 2

def denormalize(normalized_values, ranges):
    return [denormalize_value(value, ranges[i][0], ranges[i][1]) for i, value in enumerate(normalized_values)]


def plot_pareto_chart(model, feature_labels, quadratic_combinations, output_filename='pareto.pdf'):
    # Extract coefficients (excluding the intercept)
    coefficients = model.params[1:]  # Exclude the intercept
    sorted_coefficients_idx = np.argsort(np.abs(coefficients), axis=0)[::-1]
    sorted_coefficients = np.abs(coefficients[sorted_coefficients_idx])

    # Create labels for quadratic terms
    quadratic_terms = [f"{feature_labels[i]}*{feature_labels[j]}" for i, j in quadratic_combinations]

    # Bar labels (linear + quadratic)
    bar_labels = feature_labels + quadratic_terms
    bar_labels = np.array(bar_labels)
    sorted_labels = bar_labels[sorted_coefficients_idx]

    # Bar plot for the magnitude of effects
    ax1 = plt.gca()
    ax1.bar(sorted_labels, sorted_coefficients, label=r'Magnitude of effects $\beta$')
    ax1.set_xticklabels(sorted_labels, rotation=45, ha="right")
    ax1.set_ylabel(r'Magnitude of effects $\beta_{i,j}$', color='blue')
    ax1.tick_params(axis='x', which='both', bottom=False)
    ax1.tick_params(axis='y', colors='blue')

    # Calculate cumulative sum and normalize to get percentages
    cumsum_percentages = np.cumsum(sorted_coefficients) / sorted_coefficients.sum() * 100

    # Line plot for cumulative percentages on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Percentage [\%]', color='red')
    ax2.plot(sorted_labels, cumsum_percentages, color='red', linestyle='--', marker='o', label='Cumulative Percentage')
    ax2.tick_params(axis='y', colors='red')
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylim(60, 100)

    # Save the plot as a PDF
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')


# # 
# if df_normalized is not None and df_denormalized is not None:
#     if os.path.exists(csv_directory):
#         print(f"Directory found: {csv_directory}")
#         try:
#             results = pd.read_csv(os.path.join(csv_directory, "doe_test1.csv"))
#             features_const, target_values, target_lspr, target_surplus, quadratic_combinations = features_prep(results, OBJECTIVE_FUN)
#             model, model_lspr, model_surplus = fit_the_model(features_const, target_values, target_lspr, target_surplus)

#             bounds = [(-1,1),(-1,1),(-1,1),(-1,1)]  # Normalized design space
#             result = find_result(model, model_lspr, model_surplus, bounds)
            
#             ranges = [8, 80, 2, 20, 8, 80, 20, 200]  # Real-world design ranges
#             denormalized_values = denormalize(result.x, ranges)
            
#             print("Optimized Normalized Values:", result.x)
#             print("Optimized Real Values:", denormalized_values)
#         except Exception as e:
#             print(f"Error processing optimization: {e}")
#     else:
#         print(f"Error: Directory {csv_directory} does not exist.")
# else:
#     print("Could not load Excel files. Check file paths and try again.")
