import requests
import json
import os
from datetime import datetime
import subprocess

# Replace with your OpenWeatherMap API key
API_KEY = "ae30a60fc46b09faed2467c50b6c803b"

# Rawalpindi latitude and longitude
CITY_LAT = 33.6844  # Latitude of Rawalpindi
CITY_LON = 73.0479  # Longitude of Rawalpindi

# API endpoint
URL = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={CITY_LAT}&lon={CITY_LON}&appid={API_KEY}"

def run_dvc(filepath):
    """
    Adds a file to DVC and pushes it to the remote storage.
    """
    try:
        dvc_path = '/home/osama/.local/bin/dvc'

        # Add the file to DVC
        print(f"Adding {filepath} to DVC...")
        subprocess.run([dvc_path, "add", filepath], check=True)
        print(f"File {filepath} added to DVC.")

        # Push the file to DVC remote storage
        print("Pushing changes to DVC remote...")
        subprocess.run([dvc_path, "push"], check=True)
        print("Changes pushed to DVC remote.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running DVC commands: {e}")
        exit(1)

def fetch_data():
    """
    Fetches air quality data from the OpenWeatherMap API and saves it as a JSON file.
    """
    try:
        # Make the API request
        response = requests.get(URL)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Process the response
        data = response.json()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create 'data' directory if it doesn't exist
        data_dir = "/mnt/c/Users/Ahmad/Desktop/MLOPS/Project2/course-project-OsamaFahim654/enviromental_monitoring/data"
        os.makedirs(data_dir, exist_ok=True)

        # Save data to a JSON file
        filename = f"{data_dir}/air_quality_{timestamp}.json"
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Data saved to {filename}")
        
        # Add and push the file using DVC
        run_dvc(filename)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data from API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_data()
