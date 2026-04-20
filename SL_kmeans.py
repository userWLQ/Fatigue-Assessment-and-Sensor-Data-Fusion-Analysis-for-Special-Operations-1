import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


def extract_single_sheet(file_path, sheet_index):
    """Extract data from a specific sheet of an Excel file."""
    excel_file = pd.ExcelFile(file_path)
    data = pd.read_excel(file_path, sheet_name=excel_file.sheet_names[sheet_index], header=None)

    # Ensure all values are numeric; replace non-numeric with NaN and then fill with 0
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    return data.to_numpy()


def pad_or_crop_matrix(matrix, target_shape):
    """Pad or crop a matrix to match the target shape."""
    padded_matrix = np.zeros(target_shape)
    min_rows = min(matrix.shape[0], target_shape[0])
    min_cols = min(matrix.shape[1], target_shape[1])
    padded_matrix[:min_rows, :min_cols] = matrix[:min_rows, :min_cols]
    return padded_matrix


def process_all_sheets(file_paths, sheet_indices, n_clusters=3, target_shape=None):
    """Process all sheets for each file and perform overall clustering."""
    # Collect combined data for each file
    all_combined_data = []
    for file_path in file_paths:
        combined_data = []
        for sheet_index in sheet_indices:
            data = extract_single_sheet(file_path, sheet_index)
            if target_shape:
                data = pad_or_crop_matrix(data, target_shape)  # Standardize matrix size
            combined_data.append(data.flatten())  # Flatten each sheet's data
        all_combined_data.append(np.hstack(combined_data))  # Combine all sheets' data horizontally

    # Stack data into a matrix
    combined_data_matrix = np.vstack(all_combined_data)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(combined_data_matrix)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(similarity_matrix)

    return similarity_matrix, cluster_labels


def group_files_by_cluster(file_paths, cluster_labels):
    """Group files by their cluster labels."""
    file_cluster_map = {}
    for i, label in enumerate(cluster_labels):
        file_name = file_paths[i]
        if label not in file_cluster_map:
            file_cluster_map[label] = []
        file_cluster_map[label].append(file_name)
    return file_cluster_map


# Example usage:
file_paths = [
    '2/detailed_results.xlsx',
    '4/detailed_results.xlsx',
    '5/detailed_results.xlsx',
    '6/detailed_results.xlsx',
    '7/detailed_results.xlsx',
    '8（顺序变动，腰，肩，大腿，小腿）/detailed_results.xlsx',
    '10/detailed_results.xlsx',
    
    '14/detailed_results.xlsx',
    '18/detailed_results.xlsx',
    '20/detailed_results.xlsx',
    '21/detailed_results.xlsx',
    '22/detailed_results.xlsx',
    '23/detailed_results.xlsx',
    '24/detailed_results.xlsx',
    '25/detailed_results.xlsx',
    '26/detailed_results.xlsx',
    '27/detailed_results.xlsx',
    '28/detailed_results.xlsx',
    '29/detailed_results.xlsx',
    '30/detailed_results.xlsx',
    '31/detailed_results.xlsx',
    '32/detailed_results.xlsx',
    '33/detailed_results.xlsx',
    '34/detailed_results.xlsx',
    '35/detailed_results.xlsx',
    '36/detailed_results.xlsx',
]
n_clusters = 4  # Number of clusters
sheet_indices = list(range(12))  # Indices for sheets 1 to 12

# Determine the target shape based on the largest matrix
max_rows, max_cols = 0, 0
for file_path in file_paths:
    for sheet_index in sheet_indices:
        data = extract_single_sheet(file_path, sheet_index)
        max_rows = max(max_rows, data.shape[0])
        max_cols = max(max_cols, data.shape[1])

target_shape = (max_rows, max_cols)

# Process all sheets and perform overall clustering
similarity_matrix, cluster_labels = process_all_sheets(file_paths, sheet_indices, n_clusters, target_shape)
file_cluster_map = group_files_by_cluster(file_paths, cluster_labels)

# Output the overall classification results
output_data = []
for cluster, files in file_cluster_map.items():
    for file in files:
        output_data.append({"Cluster": cluster, "File": file})

# Save the results to an Excel file
output_df = pd.DataFrame(output_data)
output_df.to_excel("SL_clustering_results.xlsx", index=False)

print("Clustering results have been saved to 'clustering_results.xlsx'.")
