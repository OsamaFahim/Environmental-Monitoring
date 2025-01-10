import pandas as pd
import json
import os

def preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Example preprocessing (customize as needed)
    df = pd.json_normalize(data['list'])
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['dt'], unit='s')
    return df

def preprocess_files_in_directory(directory):
    combined_df = pd.DataFrame()  # Empty DataFrame to hold combined data
    
    for filename in os.listdir(directory):
        if filename.startswith('air_quality') and filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}")
            df = preprocess_data(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Define the full output path
    output_file_path = "/mnt/c/Users/Ahmad/Desktop/MLOPS/Project2/course-project-OsamaFahim654/enviromental_monitoring/data/combined_preprocessed_data.csv"
    
    # Remove the previous file if it exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    # Save the combined DataFrame to a single CSV file
    combined_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    preprocess_files_in_directory("/mnt/c/Users/Ahmad/Desktop/MLOPS/Project2/course-project-OsamaFahim654/enviromental_monitoring/data")
