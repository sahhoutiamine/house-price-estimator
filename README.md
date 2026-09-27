# House Price Estimator 

An end-to-end Machine Learning project to estimate house sale prices based on various characteristics (e.g., area, rooms, location, quality, build year). This project uses the popular **House Prices - Advanced Regression Techniques** dataset (Ames Housing dataset) and features a fully dockerized Streamlit web application for interactive predictions.

##  Features

- **Exploratory Data Analysis (EDA) & Modeling**: Comprehensive Jupyter notebooks covering data cleaning, feature engineering, and model training/tuning.
- **Robust Preprocessing Pipeline**: Reusable Python scripts that accurately recreate the training environment's feature vectors for real-time inference.
- **Interactive Web App**: A user-friendly Streamlit interface that accepts user input, processes it, and predicts the house price on the fly, visualizing it against historical distributions.
- **Containerization**: Fully Dockerized application for simple setup and deployment without local dependency issues.

##  Project Structure

```text
house-price-estimator/
├── app/                  # Streamlit application
│   └── streamlit_app.py  # Main entry point for the web app
├── data/                 
│   ├── raw/              # Raw data (train.csv, test.csv)
│   └── processed/        # Cleaned and processed datasets
├── models/               # Saved model artifacts (model, scaler, column references, default values)
├── notebooks/            # Jupyter Notebooks for EDA, feature engineering, and model selection
├── reports/              # Generated analysis reports and figures
├── src/                  # Source code for shared logic
│   └── preprocessing.py  # Feature engineering and data alignment logic for inference
├── Dockerfile            # Docker configuration for the application
├── docker-compose.yml    # Docker Compose setup for easy orchestration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (this file)
```

##  Setup and Installation

### Option 1: Using Docker (Recommended)

Running the app via Docker is the easiest way to get started.

1. **Build the Docker Image:**
   ```bash
   docker compose build
   ```
2. **Run the Container:**
   ```bash
   docker compose up
   ```
3. **Access the App:**
   Open your browser and navigate to http://localhost:8501
4. **Stop the Container:**
   ```bash
   docker compose down
   ```

*Note: The `models/` directory is mounted into the container. If you retrain models using the notebooks, the updated `.pkl` files will be automatically accessible to the app on restart.*

### Option 2: Local Setup (Without Docker)

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit App:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
4. **(Optional) Run Jupyter Notebooks:**
   ```bash
   jupyter notebook notebooks/
   ```

##  Machine Learning Pipeline

1. **Data Prep**: Data is sourced from `data/raw/` (ensure you place `train.csv` here if not present).
2. **Training**: Executed via the provided notebooks, generating cleaned datasets in `data/processed/` and saving artifacts in the `models/` directory using `joblib`.
3. **Inference**: The Streamlit app takes user input, applies the exact same transformations (via `src/preprocessing.py`), scales continuous variables, and outputs the prediction.

##  Tech Stack
- **Python 3.11**
- **Pandas & NumPy** (Data manipulation)
- **Scikit-Learn & XGBoost** (Modeling & Pipeline)
- **Streamlit** (Web Interface)
- **Matplotlib & Seaborn** (Data Visualization)
- **Docker** (Containerization)
