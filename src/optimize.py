import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import optimization_result_crf as optimization_result
from opt_fun_to_excel_crf import calculate_opt_fun

# Set parameters
output_directory = r'C:\Users\Lenovo\Documents\python_projects\thesis\project\doe_output_csv_5_levels2802'
normalized_file = r'C:\Users\Lenovo\Documents\python_projects\thesis\project\doe_new_5_levels.xlsx'
denormalized_file = r'C:\Users\Lenovo\Documents\python_projects\thesis\project\doe_new_5_levels_denormalized.xlsx'
factors = [8, 80, 2, 20, 8, 80, 20, 200]
bounds = [(-1, 1)] * 4
ranges = [(factors[i], factors[i + 1]) for i in range(0, len(factors), 2)]
energy_delivered_annual = 4200  # Fixed kWh/year
C_H2 = 17.40  # $/kg, hydrogen cost from study
alpha_values = np.arange(0, 1.01, 0.1)

# Debugging: Print initial parameters
print("=== Debugging: Initial Parameters ===")
print(f"Output directory: {output_directory}")
print(f"Normalized file: {normalized_file}")
print(f"Denormalized file: {denormalized_file}")
print(f"Factors: {factors}")
print(f"Bounds: {bounds}")
print(f"Ranges: {ranges}")
print(f"Energy_delivered_annual: {energy_delivered_annual}")
print(f"C_H2: {C_H2}")
print(f"Alpha values for F1: {alpha_values}")

# Load designs
try:
    df_normalized = pd.read_excel(normalized_file)
    df_normalized = (df_normalized - 2) / 2  # Normalize coded levels [0, 4] to [-1, 1]
    print("Debug: Unique values in df_normalized:\n", np.unique(df_normalized.to_numpy(), axis=0))
    df_denormalized = pd.read_excel(denormalized_file)
    print("\n=== Debugging: Loaded Excel Files ===")
    print(f"df_normalized shape: {df_normalized.shape}")
    print(f"df_denormalized shape: {df_denormalized.shape}")
except Exception as e:
    print(f"Error reading Excel files: {e}")
    exit()

# Compute optimization components
df_functions, dropped_files = calculate_opt_fun(output_directory, df_normalized, df_denormalized)
print(df_functions)
if df_functions is None:
    raise ValueError("calculate_opt_fun failed. Check CSV files or error messages above.")
if dropped_files:
    if len(dropped_files) > 3:
        print(f"Dropping last row of files {dropped_files[0]} to {dropped_files[-1]}")
    else:
        print(f"Dropping last row of files: {', '.join(dropped_files)}")
print("\n=== Debugging: df_functions ===")
print(f"df_functions shape: {df_functions.shape}")
print(f"df_functions columns: {df_functions.columns.tolist()}")
print(f"NaN in df_functions:\n{df_functions.isna().sum()}")
print(f"total_costs stats:\n{df_functions['total_costs'].describe()}")

# Save df_functions to Excel for inspection
df_functions.to_excel('df_functions_output.xlsx', index=False)
print("\n=== Saved df_functions to 'df_functions_output.xlsx' ===")

# Compute min and max for total_costs normalization
total_costs_min = df_functions['total_costs'].min()
total_costs_max = df_functions['total_costs'].max()

print(f"\n=== Debugging: total_costs Normalization ===")
print(f"total_costs min: {total_costs_min:.2f}, max: {total_costs_max:.2f}")

# Compute hydrogen cost term
df_functions['hydrogen_cost_term'] = (C_H2 / 33.3) * (df_functions['energy_deficit'] + df_functions['energy_loss'])
h2_cost_min = df_functions['hydrogen_cost_term'].min()
h2_cost_max = df_functions['hydrogen_cost_term'].max()
print("\n=== Debugging: Hydrogen Cost Term ===")
print(f"hydrogen_cost_term stats: min={df_functions['hydrogen_cost_term'].min():.4f}, "
      f"max={df_functions['hydrogen_cost_term'].max():.4f}, "
      f"mean={df_functions['hydrogen_cost_term'].mean():.4f}")

# Prepare and fit models for each component
targets = ['total_costs', 'LPSR', 'hydrogen_cost_term']
models = {}
r2_scores = {}
predictions = {}
poly_transformers = {}

for target in targets:
    print(f"\n=== Debugging: Preparing Features for {target} ===")
    features_poly, target_values, poly = optimization_result.features_prep(df_functions, target)
    print(f"Features shape: {features_poly.shape}")
    print(f"Target values shape: {target_values.shape}")
    model = optimization_result.fit_the_model(features_poly, target_values)
    models[target] = model
    poly_transformers[target] = poly
    y_pred = model.predict(features_poly)
    predictions[target] = y_pred
    r2_scores[target] = r2_score(target_values, y_pred)
    print(f"R² Score for {target}: {r2_scores[target]:.4f}")

# Check predictions for key sizes
print("\n=== Debugging: Predictions for Key Sizes ===")
key_sizes = [
    [80, 2, 8, 200],
    [80, 2, 8, 20],
    [8, 2, 8, 20]
]
for sizes in key_sizes:
    norm_sizes = [(s - r[0]) / (r[1] - r[0]) * 2 - 1 for s, r in zip(sizes, ranges)]  # Normalize to [-1, 1]
    df_x = pd.DataFrame([norm_sizes], columns=['PV', 'Battery', 'SOFC', 'TANK'])
    
    # Prepare features for all targets
    features_poly, _, poly = optimization_result.features_prep(df_x, None)
    total_costs_pred = models['total_costs'].predict(features_poly)[0]
    lpsr_pred = models['LPSR'].predict(features_poly)[0]
    h2_cost_pred = models['hydrogen_cost_term'].predict(features_poly)[0]
    
    print(f"Sizes {sizes}: total_costs_pred={total_costs_pred:.2f}, LPSR_pred={lpsr_pred:.4f}, h2_cost_pred={h2_cost_pred:.2f}")

# # Plot histograms of df_functions
# print("\n=== Plotting Histograms ===")
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for ax, target in zip(axes, targets):
#     ax.hist(df_functions[target], bins=30, edgecolor='black')
#     ax.set_title(f'Histogram of {target}')
#     ax.set_xlabel(target)
#     ax.set_ylabel('Frequency')
# plt.tight_layout()
# plt.show()

from sklearn.model_selection import train_test_split

targets = ['total_costs', 'LPSR', 'hydrogen_cost_term']
models = {}
r2_scores = {}
predictions = {}
poly_transformers = {}

print("\n=== Debugging: Model Fitting with Train-Test Split ===")
for target in targets:
    print(f"\n=== Preparing Features for {target} ===")
    features_poly, target_values, poly = optimization_result.features_prep(df_functions, target)
    print(f"Features shape: {features_poly.shape}")
    print(f"Target values shape: {target_values.shape}")

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        features_poly, target_values, test_size=0.2, random_state=0
    )
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Fit model on training data
    model = optimization_result.fit_the_model(X_train, y_train)
    models[target] = model
    poly_transformers[target] = poly

    # Evaluate on test data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_scores[target] = {
        'train': r2_score(y_train, y_pred_train),
        'test': r2_score(y_test, y_pred_test)
    }
    predictions[target] = {
        'train': y_pred_train,
        'test': y_pred_test,
        'train_x': X_train,
        'test_x': X_test,
        'train_true': y_train,
        'test_true': y_test
    }
    print(f"R² Score for {target}: Train={r2_scores[target]['train']:.4f}, Test={r2_scores[target]['test']:.4f}")

# Plot true vs predicted scatter plots with train-test distinction
print("\n=== Plotting True vs Predicted Scatter Plots (Train vs Test) ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, target in zip(axes, targets):
    # Plot training data
    ax.scatter(predictions[target]['train_true'], predictions[target]['train'], 
               c='blue', label='Train', alpha=0.5)
    # Plot test data
    ax.scatter(predictions[target]['test_true'], predictions[target]['test'], 
               c='red', marker='s', label='Test', alpha=0.5)
    # Diagonal line
    min_val = df_functions[target].min()
    max_val = df_functions[target].max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax.set_title(f'True vs Predicted {target} (R² Test: {r2_scores[target]["test"]:.4f})')
    ax.set_xlabel(f'True {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()

# # Plot true vs predicted scatter plots
# # Save true vs predicted scatter plots to PDF
# print("\n=== Saving True vs Predicted Scatter Plots to PDF ===")
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for ax, target in zip(axes, targets):
#     ax.scatter(df_functions[target], predictions[target], alpha=0.5)
#     ax.plot([df_functions[target].min(), df_functions[target].max()],
#             [df_functions[target].min(), df_functions[target].max()], 'r--')
#     ax.set_title(f'True vs Predicted {target}')
#     ax.set_xlabel(f'True {target}')
#     ax.set_ylabel(f'Predicted {target}')
#     ax.grid(True)
# plt.tight_layout()
# fig.savefig('true_vs_predicted_scatter_plots.pdf', format='pdf', bbox_inches='tight')
# plt.close(fig)  # Close the figure to free memory

# Search for optimal alpha values for F1
print("\n=== Debugging: Alpha Search for F1 ===")
results_f1 = []
for alpha in alpha_values:
    print(f"\n=== Optimizing for alpha={alpha:.3f} ===")
    weights_f1 = {
        'total_costs': alpha,
        'lpsr': 1 - alpha,
        'h2_cost': 0
    }
    print(f"Weights: total_costs={alpha:.3f}, lpsr={1-alpha:.3f}, h2_cost=0")
    result_f1 = optimization_result.find_result(
        models['total_costs'], models['LPSR'], models['hydrogen_cost_term'],
        poly_transformers['total_costs'], poly_transformers['LPSR'], poly_transformers['hydrogen_cost_term'],
        bounds, weights_f1, total_costs_min, total_costs_max, h2_cost_min, h2_cost_max
    )
    denormalized_values = optimization_result.denormalize(result_f1.x, ranges)
    optimized_sizes = np.ceil(denormalized_values)
    print(f"Optimized normalized values: {result_f1.x}")
    print(f"Denormalized values: {denormalized_values}")
    print(f"Optimized sizes: {optimized_sizes}")
    print(f"Optimization success: {result_f1.success}, message: {result_f1.message}")

    # Predict total_costs and LPSR
    df_x = pd.DataFrame([result_f1.x], columns=['PV', 'Battery', 'SOFC', 'TANK'])
    features_poly, _, _ = optimization_result.features_prep(df_x, None)
    total_costs_pred = models['total_costs'].predict(features_poly)[0]
    lpsr_pred = models['LPSR'].predict(features_poly)[0]
    lcoe_pred = total_costs_pred / energy_delivered_annual / 30
    print(f"Predicted total_costs: {total_costs_pred:.2f}")
    print(f"Predicted LPSR: {lpsr_pred:.4f}")
    print(f"Predicted LCOE: {lcoe_pred:.4f}")
    results_f1.append({
        'alpha': alpha,
        'total_costs': total_costs_pred,
        'LPSR': lpsr_pred,
        'optimized_sizes': optimized_sizes,
        'LCOE': lcoe_pred
    })

# Plot Pareto front: LPSR vs total_costs
print("\n=== Plotting Pareto Front for F1 ===")
total_costs = [res['total_costs'] for res in results_f1]
lpsr = [res['LPSR'] for res in results_f1]
plt.figure(figsize=(10, 6))
plt.scatter(lpsr, total_costs, c=alpha_values, cmap='viridis', s=100)
plt.colorbar(label='Alpha')
plt.xlabel('LPSR')
plt.ylabel('Total Costs ($)')
plt.title('Pareto Front: LPSR vs Total Costs (F1)')
plt.grid(True)
for i, res in enumerate(results_f1):
    if i % 2 == 0:  # Annotate every other point to avoid clutter
        plt.annotate(f"α={res['alpha']:.3f}", (lpsr[i], total_costs[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')
plt.tight_layout()
plt.show()

# Print results table
print("\n=== F1 Optimization Results Table ===")
print("Alpha | Total Costs ($) | LPSR | LCOE ($/kWh) | Optimized Sizes")
print("-" * 70)
for res in results_f1:
    print(f"{res['alpha']:.3f} | {res['total_costs']:.2f} | {res['LPSR']:.4f} | {res['LCOE']:.4f} | {res['optimized_sizes']}")

# Search for optimal alpha values for F1
print("\n=== Debugging: Alpha Search for F1 ===")
results_f1 = []
for alpha in alpha_values:
    print(f"\n=== Optimizing for alpha={alpha:.3f} ===")
    weights_f1 = {
        'total_costs': alpha,
        'lpsr': 1 - alpha,
        'h2_cost': 0
    }
    print(f"Weights: total_costs={alpha:.3f}, lpsr={1-alpha:.3f}, h2_cost=0")
    result_f1 = optimization_result.find_result(
        models['total_costs'], models['LPSR'], models['hydrogen_cost_term'],
        poly_transformers['total_costs'], poly_transformers['LPSR'], poly_transformers['hydrogen_cost_term'],
        bounds, weights_f1, total_costs_min, total_costs_max, h2_cost_min, h2_cost_max
    )
    denormalized_values = optimization_result.denormalize(result_f1.x, ranges)
    optimized_sizes = np.ceil(denormalized_values)
    print(f"Optimized normalized values: {result_f1.x}")
    print(f"Denormalized values: {denormalized_values}")
    print(f"Optimized sizes: {optimized_sizes}")
    print(f"Optimization success: {result_f1.success}, message: {result_f1.message}")

    # Predict total_costs, LPSR, and h2_cost
    df_x = pd.DataFrame([result_f1.x], columns=['PV', 'Battery', 'SOFC', 'TANK'])
    features_poly, _, _ = optimization_result.features_prep(df_x, None)
    total_costs_pred = models['total_costs'].predict(features_poly)[0]
    lpsr_pred = models['LPSR'].predict(features_poly)[0]
    lcoe_pred = total_costs_pred / energy_delivered_annual / 30
    print(f"Predicted total_costs: {total_costs_pred:.2f}")
    print(f"Predicted LPSR: {lpsr_pred:.4f}")
    print(f"Predicted LCOE: {lcoe_pred:.4f}")
    results_f1.append({
        'alpha': alpha,
        'total_costs': total_costs_pred,
        'LPSR': lpsr_pred,
        'optimized_sizes': optimized_sizes,
        'LCOE': lcoe_pred
    })

# Save F1 Pareto front to PDF
print("\n=== Saving F1 Pareto Front to PDF ===")
fig, ax = plt.subplots(figsize=(10, 6))
total_costs = [res['total_costs'] for res in results_f1]
lpsr = [res['LPSR'] for res in results_f1]
alpha_vals = [res['alpha'] for res in results_f1]
scatter = ax.scatter(lpsr, total_costs, c=alpha_vals, cmap='viridis', s=100, label='F1', alpha=0.6)
plt.colorbar(scatter, label='Alpha')
ax.set_xlabel('LPSR')
ax.set_ylabel('Total Costs ($)')
ax.set_title('Pareto Front: LPSR vs Total Costs (F1)')
ax.grid(True)
for i, res in enumerate(results_f1):
    if i % 2 == 0:  # Annotate every other point to avoid clutter
        ax.annotate(f"α={res['alpha']:.3f}", (lpsr[i], total_costs[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')
plt.tight_layout()
fig.savefig('pareto_front_f1.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)  # Close the figure to free memory


# Search for optimal alpha and beta values for F2 with adjusted granularity
print("\n=== Debugging: Alpha and Beta Search for F2 ===")
results_f2 = []  

# Coarse region (0.6 <= alpha + beta <= 0.8)
# alpha_coarse = [0.0, 0.3, 0.6]
# beta_coarse = [0.6, 0.7, 0.8]# Fine region (0.9 <= alpha + beta <= 1.0)
# alpha_fine = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
# beta_fine = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]# Transition buffer
# alpha_buffer = [0.7, 0.8]
# beta_buffer = [0.1, 0.2]

def generate_simplex_grid(n):
    """Generate uniformly spaced barycentric coordinates on the 2-simplex."""
    points = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            a1 = i / n
            a2 = j / n
            a3 = k / n
            points.append([a1, a2, a3])
    return np.array(points)


for alpha, beta, h2_weight in generate_simplex_grid(10):
    h2_weight = 1 - alpha - beta
    print(f"\n=== Optimizing for alpha={alpha:.3f}, beta={beta:.3f}, h2_weight={h2_weight:.3f} ===")
    weights_f2 = {
        'total_costs': alpha,
        'lpsr': beta,
        'h2_cost': h2_weight
    }
    print(f"Weights: total_costs={alpha:.3f}, lpsr={beta:.3f}, h2_cost={h2_weight:.3f}")
    result_f2 = optimization_result.find_result(
        models['total_costs'], models['LPSR'], models['hydrogen_cost_term'],
        poly_transformers['total_costs'], poly_transformers['LPSR'], poly_transformers['hydrogen_cost_term'],
        bounds, weights_f2, total_costs_min, total_costs_max, h2_cost_min, h2_cost_max
    )
    denormalized_values = optimization_result.denormalize(result_f2.x, ranges)
    optimized_sizes = np.ceil(denormalized_values)
    print(f"Optimized normalized values: {result_f2.x}")
    print(f"Denormalized values: {denormalized_values}")
    print(f"Optimized sizes: {optimized_sizes}")
    print(f"Optimization success: {result_f2.success}, message: {result_f2.message}")    # Predict total_costs, LPSR, and h2_cost
    df_x = pd.DataFrame([result_f2.x], columns=['PV', 'Battery', 'SOFC', 'TANK'])
    features_poly, _, _ = optimization_result.features_prep(df_x, None)
    total_costs_pred = models['total_costs'].predict(features_poly)[0]
    lpsr_pred = models['LPSR'].predict(features_poly)[0]
    h2_cost_pred = models['hydrogen_cost_term'].predict(features_poly)[0]
    lcoe_pred = total_costs_pred / energy_delivered_annual / 30
    print(f"Predicted total_costs: {total_costs_pred:.2f}")
    print(f"Predicted LPSR: {lpsr_pred:.4f}")
    print(f"Predicted h2_cost: {h2_cost_pred:.2f}")
    print(f"Predicted LCOE: {lcoe_pred:.4f}")
    results_f2.append({
        'alpha': alpha,
        'beta': beta,
        'h2_weight': h2_weight,
        'total_costs': total_costs_pred,
        'LPSR': lpsr_pred,
        'h2_cost': h2_cost_pred,
        'optimized_sizes': optimized_sizes,
        'LCOE': lcoe_pred,
        'PV': result_f2.x[0],
        'Battery': result_f2.x[1],
        'SOFC': result_f2.x[2],
        'TANK': result_f2.x[3]
    })
    
# Save Pareto front to PDF
print("\n=== Saving Pareto Front for F1 and F2 to PDF ===")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))# Plot F1 (commented out, using viridis for alpha, to change later)
# total_costs_f1 = [res['total_costs'] for res in results_f1]
# lpsr_f1 = [res['LPSR'] for res in results_f1]
# alpha_f1 = [res['alpha'] for res in results_f1]
# ax.scatter(lpsr_f1, total_costs_f1, c=alpha_f1, cmap='viridis', s=100, label='F1 (Alpha)', alpha=0.6)
# plt.colorbar(label='Alpha (F1)')
# ax.legend()



# Plot F2 with size for beta and edge width for alpha
total_costs_f2 = [res['total_costs'] for res in results_f2]
print(total_costs_f2)
lpsr_f2 = [res['LPSR'] for res in results_f2]
h2_cost_f2 = [res['h2_cost'] for res in results_f2]
alpha_f2 = [res['alpha'] for res in results_f2]
beta_f2 = [res['beta'] for res in results_f2]  # Scale size for beta (50 to 500 based on beta range 0 to 1)
pv_f2 = [res['PV'] for res in results_f2]
battery_f2 = [res['Battery'] for res in results_f2]
sofc_f2 = [res['SOFC'] for res in results_f2]
tank_f2 = [res['TANK'] for res in results_f2]

pd.DataFrame({'total_costs_f2': total_costs_f2, 
              'lpsr_f2': lpsr_f2,
              'h2_cost_f2': h2_cost_f2,
              'alpha_f2': alpha_f2,
            'beta_f2': beta_f2,
            'pv_f2': pv_f2,
            'battery_f2': battery_f2,
            'sofc_f2': sofc_f2,
            'tank_f2': tank_f2
              }).to_csv('res_f2.csv')


sizes = [50 + 450 * beta for beta in beta_f2]  # Scale edge width for alpha (0.5 to 3.0 based on alpha range 0 to 1)
edge_widths = [0.5 + 2.5 * alpha for alpha in alpha_f2]  # Plot with consistent marker (circle) and vary size and edge width
scatter_f2 = ax.scatter(lpsr_f2, h2_cost_f2, s=total_costs_f2/np.max(total_costs_f2)*100, edgecolors='black', 
                        linewidths=edge_widths, facecolors='none', label='F2', alpha=0.6)  # Add legend for beta (size) and alpha (edge width)
beta_levels = [0.0, 0.5, 1.0]  # Representative levels
size_samples = [50 + 450 * b for b in beta_levels]
alpha_levels = [0.0, 0.5, 1.0]  # Representative levels
width_samples = [0.5 + 2.5 * a for a in alpha_levels]
from matplotlib.lines import Line2D
beta_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=np.sqrt(s/10), markeredgecolor='black', 
                      markeredgewidth=1.5, label=f'β={b:.1f}') for s, b in zip(size_samples, beta_levels)]
alpha_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                       markersize=10, markeredgecolor='black', 
                       markeredgewidth=w, label=f'α={a:.1f}') for w, a in zip(width_samples, alpha_levels)]
ax.legend(handles=beta_legend + alpha_legend, title='F2: Beta (Size), Alpha (Edge Width)', 
          loc='best', ncol=2)
ax.set_title('F2 Pareto Front (Beta by Size, Alpha by Edge Width)')
ax.set_xlabel('h2_cost_f2')
ax.set_ylabel('LPSR')
ax.grid(True)
plt.tight_layout()
fig.savefig('pareto_front_f1_f2.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)  # Close the figure to free memory

