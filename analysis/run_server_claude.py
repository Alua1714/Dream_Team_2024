# Configuration for model storage
import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# 1. Set up MLflow storage configuration
class MLFlowConfig:
    def __init__(self, storage_path="./mlruns"):
        # Local storage path for MLflow artifacts
        self.storage_path = storage_path
        # Set tracking URI - could be local or remote
        mlflow.set_tracking_uri(f"file://{os.path.abspath(storage_path)}")
        self.client = MlflowClient()
        
    def save_production_model(self, run_id, model_name):
        """Register the model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/model"
        
        # Register model if it doesn't exist
        try:
            version = mlflow.register_model(model_uri, model_name)
            print(f"Registered model version: {version.version}")
        except Exception as e:
            print(f"Model already exists, creating new version: {str(e)}")
            version = mlflow.register_model(model_uri, model_name)
        
        # Set model version as production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Production"
        )
        
        return version

# 2. Example usage in your training script
def train_and_save_model():
    mlflow_config = MLFlowConfig()
    
    with mlflow.start_run() as run:
        # Your model training code here (using previous example)
        model = RandomForestRegressor(**sklearn_params)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Register model in production
        version = mlflow_config.save_production_model(
            run.info.run_id, 
            "house_price_predictor"
        )
        
        return run.info.run_id, version

# 3. Flask server for model deployment
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the production model
def load_production_model(model_name):
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/Production"
    )
    return model

# Initialize model
MODEL_NAME = "house_price_predictor"
model = load_production_model(MODEL_NAME)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from request
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)
        
        return jsonify({
            'prediction': prediction.tolist()[0],
            'model_version': mlflow.pyfunc.get_model_info(model.model_uri).version
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 4. Docker deployment
# Save as Dockerfile
"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""

# 5. Requirements.txt
"""
mlflow
flask
gunicorn
pandas
scikit-learn
lightgbm
"""

# 6. Example of model reloading in production
def reload_production_model():
    global model
    model = load_production_model(MODEL_NAME)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)