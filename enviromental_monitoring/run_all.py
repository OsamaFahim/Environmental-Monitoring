import os
import subprocess

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(['python3', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        exit(1)

if __name__ == "__main__":
    # Set the base directory
    BASE_DIR = "/mnt/c/Users/Ahmad/Desktop/MLOPS/Project2/course-project-OsamaFahim654/enviromental_monitoring"

    # Change directory to the base directory
    os.chdir(BASE_DIR)

    # Run the data collection, preprocessing scripts
    run_script(os.path.join(BASE_DIR, 'Data_collections.py'))
    run_script(os.path.join(BASE_DIR, 'data_preprocessing.py'))

    # Finally run the train_model.py
    run_script(os.path.join(BASE_DIR, 'train_model.py'))

    print("All tasks completed successfully.")
