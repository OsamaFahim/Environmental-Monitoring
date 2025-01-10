
# Environmental Monitoring and Pollution Prediction System

The **Environmental Monitoring and Pollution Prediction System** is an MLOps pipeline designed to monitor real-time environmental data, predict pollution trends, and alert users about high-risk pollution days. It integrates **DVC**, **MLflow**, **FastAPI**, and **Grafana** for data management, model tracking, API deployment, and system monitoring.

---

## Features
1. **Real-Time Environmental Data**:  
   Fetches live weather and air quality data from APIs like OpenWeatherMap, AirVisual, and more.
2. **Data Versioning**:  
   Tracks changes in environmental data using DVC and remote storage integration.
3. **Pollution Trend Prediction**:  
   Uses time-series models (ARIMA, LSTM) to forecast pollution levels.
4. **Deployed Prediction API**:  
   Predicts pollution trends via a FastAPI server.
5. **Real-Time Monitoring**:  
   Tracks API and model performance with Grafana and Prometheus.

---

## Technologies Used
- **Backend**: Python with FastAPI  
- **Data Management**: DVC (Data Version Control)  
- **Model Tracking**: MLflow  
- **Monitoring**: Prometheus and Grafana  
- **Database**: SQLite or MySQL  
- **Task Automation**: Cron jobs for scheduled data fetching  

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/OsamaFahim/Environmental-Monitoring.git
cd Environmental-Monitoring
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up DVC
- Initialize the DVC repository:
  ```bash
  dvc init
  ```
- Add remote storage (e.g., Google Drive):
  ```bash
  dvc remote add -d myremote <remote-url>
  ```
- Push data to remote storage:
  ```bash
  dvc push
  ```

### 4. Configure Data Fetching
- Add API keys for external services (e.g., OpenWeatherMap) in a `.env` file:
  ```bash
  API_KEY_OPENWEATHER=<your_openweather_api_key>
  API_KEY_AIRVISUAL=<your_airvisual_api_key>
  ```

- Schedule a **cron job** to automate data fetching:
  ```bash
  crontab -e
  ```
  Add this line to fetch data every hour:
  ```bash
  0 * * * * /usr/bin/python3 /path/to/fetch_data.py
  ```

### 5. Run the FastAPI Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
- Access the FastAPI docs at [http://localhost:8000/docs](http://localhost:8000/docs).  

### 6. Start MLflow
```bash
mlflow ui
```
- View MLflow Tracking UI at [http://localhost:5050](http://localhost:5050).  

### 7. Configure Grafana and Prometheus
- Use Docker Compose or a manual setup to run Grafana and Prometheus.  
- Visualize real-time data ingestion, model predictions, and API performance.

---

## Usage

### Data Fetching
Run the data collection script manually or via cron:
```bash
python fetch_data.py
```

### Model Training
Train models and log experiments using MLflow:
```bash
python train_model.py
```

### API for Predictions
Make predictions by sending requests to the FastAPI endpoint:
```bash
POST http://localhost:8000/predict
```

### Real-Time Monitoring
Access the Grafana dashboard to view:
- Data ingestion metrics
- Model prediction trends
- API performance statistics  

---

## Folder Structure
```
Environmental-Monitoring/
│
├── data/                 # Environmental data managed by DVC
├── scripts/              # Scripts for data fetching and preprocessing
├── models/               # Trained models and logs (via MLflow)
├── dashboards/           # Grafana dashboard configurations
├── api.py                # FastAPI implementation for predictions
├── fetch_data.py         # Script to fetch and store data
├── train_model.py        # Script for model training
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## Contributing
We welcome contributions! Please follow these steps:  
1. Fork the repository.  
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit and push your changes:
   ```bash
   git commit -m "Add feature-name"
   git push origin feature-name
   ```
4. Open a pull request.

---

## License
This project is licensed under the **MIT License**.

---

## Contact
For any questions or issues, please open an issue in the repository or contact the project maintainers.
