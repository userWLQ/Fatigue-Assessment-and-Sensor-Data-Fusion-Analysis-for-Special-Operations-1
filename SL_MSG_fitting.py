import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import random
import os
from plotting_style import SCI_COLORS, setup_sci_style

setup_sci_style()


def convert_txt_to_excel(name):
    """
    Convert TXT data file to Excel format
    
    Parameters:
    - name: folder name containing the data file
    
    Returns:
    - Path to the converted Excel file
    """
    # Load the data file into a DataFrame
    file_path = name + '/MVC.txt'
    df = pd.read_csv(file_path, sep="\t")
    
    # Export all columns into a single Excel sheet
    output_path_single_sheet = name + '/MVC_single_sheet.xlsx'
    df.to_excel(output_path_single_sheet, index=False)
    
    return output_path_single_sheet


def calculate_global_standard_value_from_averages(data, chunk_size=500):
    """
    Calculate global standard value from data averages
    
    Parameters:
    - data: input data array
    - chunk_size: size of chunks to average
    
    Returns:
    - Calculated standard value
    """
    num_chunks = len(data) // chunk_size
    averaged_data = [np.mean(data[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]
    averaged_data = np.array(averaged_data)
    values, counts = np.unique(averaged_data, return_counts=True)
    most_frequent_values = values[np.argsort(counts)[-5:]]  # Top 5 most frequent values
    return np.median(most_frequent_values)


def detect_global_deviation_regions_filtered(data, standard_value, deviation_threshold=10, min_points=5):
    """
    Detect continuous deviation regions from the standard value, keeping only regions with more than min_points.
    
    Parameters:
    - data: input data
    - standard_value: global standard value
    - deviation_threshold: deviation threshold
    - min_points: minimum number of points requirement
    
    Returns:
    - List of qualifying deviation regions
    """
    is_movement = np.abs(data - standard_value) > deviation_threshold
    deviation_regions = []
    start_idx = None
    for i, movement in enumerate(is_movement):
        if movement and start_idx is None:
            start_idx = i
        elif not movement and start_idx is not None:
            if i - start_idx > min_points:  # Only keep regions with more than min_points
                deviation_regions.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None and len(data) - start_idx > min_points:  # Handle region at the end
        deviation_regions.append((start_idx, len(data) - 1))
    return deviation_regions


def fit_polynomials_fixed(data, regions, degree=3):
    """
    Fit polynomials to deviation regions
    
    Parameters:
    - data: input data
    - regions: deviation regions to fit
    - degree: polynomial degree
    
    Returns:
    - Array of polynomial coefficients
    """
    coefficients = []
    for start, end in regions:
        region_data = data[start:end + 1]  # Include end point
        if len(region_data) <= degree:  # Skip if not enough points for fitting
            continue
        x = np.arange(len(region_data))
        try:
            poly = Polynomial.fit(x, region_data, degree)
            coef = poly.convert().coef  # Convert to power basis form
            coef = np.pad(coef, (0, degree + 1 - len(coef)), 'constant')
            coefficients.append(coef)
        except np.linalg.LinAlgError:
            continue
    return np.array(coefficients)


def construct_matrices(coefficients_list):
    """
    Construct matrices from polynomial coefficients
    
    Parameters:
    - coefficients_list: list of polynomial coefficients
    
    Returns:
    - Stacked matrix of coefficients
    """
    return np.vstack(coefficients_list)

# Plotting function 
def plot_fitted_regions_combined_save(data, regions, coefficients, num_plots=10, degree=3, output_folder="plots"):
    """
    Randomly plot fitting results for deviation regions, combine multiple fitting curves in one figure, and save to file.
    
    Parameters:
    - data: input data
    - regions: list of deviation regions
    - coefficients: fitted polynomial coefficients
    - num_plots: number of plots to generate
    - degree: polynomial degree
    - output_folder: folder to save images
    """
    import os

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Filter valid regions and corresponding coefficients
    valid_regions_and_coeffs = [
        (start, end, coef) for (start, end), coef in zip(regions, coefficients) if len(coef) == degree + 1
    ]

    if not valid_regions_and_coeffs:
        print("No valid regions or coefficients available for plotting.")
        return

    # Randomly sample regions
    sampled = random.sample(valid_regions_and_coeffs, min(num_plots, len(valid_regions_and_coeffs)))

    # Calculate subplot grid dimensions
    cols = 5
    rows = (num_plots + cols - 1) // cols  # Round up

    # Create a large figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.8))
    axes = axes.flatten()  # Flatten for easy iteration

    for i, (start, end, coef) in enumerate(sampled):
        ax = axes[i]
        region_data = data[start:end + 1]
        x = np.arange(len(region_data))
        fitted_poly = Polynomial(coef)
        fitted_y = fitted_poly(x)
        ax.plot(x, region_data, label="Original data", marker="o", color=SCI_COLORS["secondary"])
        ax.plot(x, fitted_y, label=f"Fitted polynomial ({degree} deg)", linestyle="--", color=SCI_COLORS["accent"])
        ax.set_title(f"Fitting region: Start={start}, End={end}", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide extra subplots if needed
    for j in range(len(sampled), len(axes)):
        fig.delaxes(axes[j])

    # Save to file
    plot_file_path = os.path.join(output_folder, f"fitted_regions_combined.png")
    plt.tight_layout()
    plt.savefig(plot_file_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Combined plot saved to folder: {output_folder}")

# Main processing function 
def process_uploaded_data_with_plotting_and_saving(input_file_path, output_file_path_standard_values, output_file_path_results, chunk_size=500, degree=3, plot_output_folder="plots"):
    """
    Process uploaded data, calculate standard values, detect deviations, fit polynomials, and save results
    
    Parameters:
    - input_file_path: path to input Excel file
    - output_file_path_standard_values: path to save standard values
    - output_file_path_results: path to save detailed results
    - chunk_size: size of chunks for averaging
    - degree: polynomial degree for fitting
    - plot_output_folder: folder to save plots
    """
    df_uploaded = pd.read_excel(input_file_path)

    global_standard_values = {}
    deviation_regions_results = {}
    polynomial_coefficients_results = {}
    matrix_results = {}

    for column in df_uploaded.columns:
        data_column = df_uploaded[column].dropna().to_numpy()
        if len(data_column) < chunk_size:
            global_standard_values[column] = np.nan
            continue

        standard_value = calculate_global_standard_value_from_averages(data_column, chunk_size)
        global_standard_values[column] = standard_value

        deviation_regions = detect_global_deviation_regions_filtered(data_column, standard_value)
        deviation_regions_results[column] = deviation_regions

        coefficients = fit_polynomials_fixed(data_column, deviation_regions, degree)
        polynomial_coefficients_results[column] = coefficients

        if len(coefficients) > 0:
            matrix_results[column] = construct_matrices(coefficients)
        else:
            matrix_results[column] = None

    global_standard_values_df = pd.DataFrame.from_dict(
        global_standard_values, orient="index", columns=["Global Standard Value"]
    )
    global_standard_values_df.to_excel(output_file_path_standard_values, index_label="Column")

    with pd.ExcelWriter(output_file_path_results) as writer:
        for column, regions in deviation_regions_results.items():
            regions_df = pd.DataFrame(regions, columns=["Start Index", "End Index"])
            regions_df.to_excel(writer, sheet_name=f"{column}_Deviation Regions", index=False)

        for column, coefficients in polynomial_coefficients_results.items():
            if coefficients is not None:
                coef_df = pd.DataFrame(coefficients, columns=[f"Coeff_{i}" for i in range(coefficients.shape[1])])
                coef_df.to_excel(writer, sheet_name=f"{column}_Polynomial Coeffs", index=False)

        for column, matrix in matrix_results.items():
            if matrix is not None:
                matrix_df = pd.DataFrame(matrix)
                matrix_df.to_excel(writer, sheet_name=f"{column}_Matrix", index=False)

    # Randomly plot fitting results and save to a combined image
    for column, regions in deviation_regions_results.items():
        coefficients = polynomial_coefficients_results.get(column)
        if coefficients is not None:
            data_column = df_uploaded[column].dropna().to_numpy()
            print(f"Generating combined plot for column: {column}")
            column_output_folder = f"{plot_output_folder}/{column.replace(' ', '_')}"
            plot_fitted_regions_combined_save(data_column, regions, coefficients, num_plots=10, degree=degree, output_folder=column_output_folder)

    print("All results computed and saved to specified files!")


if __name__ == "__main__":
    # Ensure the output folder exists
    name = "36"
    os.makedirs(name, exist_ok=True)

    # Step 1: Convert file
    print("Step 1: Converting data file to Excel format...")
    converted_excel_path = convert_txt_to_excel(name)
    print(f"File converted and saved to: {converted_excel_path}")

    # Step 2: Process data
    print("Step 2: Starting data processing...")
    output_standard_values = os.path.join(name, "global_standard_values.xlsx")
    output_results = os.path.join(name, "detailed_results.xlsx")
    plot_output_folder = os.path.join(name, "plots")

    process_uploaded_data_with_plotting_and_saving(
        converted_excel_path,
        output_standard_values,
        output_results,
        plot_output_folder=plot_output_folder
    )

    print("All processing completed! Results and plots have been saved.")
