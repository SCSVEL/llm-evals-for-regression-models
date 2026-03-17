"""
Main Pipeline Script for Regression Model Building
Orchestrates the complete pipeline: data ingestion -> cleaning -> modeling -> evaluation
"""

import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import DataIngestion
from data_cleaning import DataCleaner
from model_building import ModelBuilder
from evaluation import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Execute the complete regression pipeline for home price prediction."""

    logger.info("Starting Home Price Regression Model Pipeline")

    # Create directories if not exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 1. Data Ingestion
    logger.info("Step 1: Data Ingestion")
    ingestion = DataIngestion()

    # Load Zillow Home Value Index data for State College, PA
    file_path = 'data/raw/zhvi_metro.csv'
    data = ingestion.load_zillow_home_prices(file_path, region_name="State College, PA")

    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"Data date range: {data['date'].min()} to {data['date'].max()}")

    # 2. Data Cleaning
    logger.info("Step 2: Data Cleaning")
    cleaner = DataCleaner()
    cleaner.load_data(data)

    # Clean data
    cleaned_data = cleaner.clean_data()

    # Prepare features and target
    # For time series regression, use time_index as feature, price as target
    X = cleaned_data[['time_index']]
    y = cleaned_data['price']

    logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

    # 3. Model Building
    logger.info("Step 3: Model Building")
    builder = ModelBuilder()

    # Split data
    X_train, X_test, y_train, y_test = builder.prepare_data(X, y)

    # Define models
    models = builder.define_models()

    # Tune hyperparameters
    best_models = builder.tune_hyperparameters(X_train, y_train)

    # Train final model
    final_model = builder.train_final_model(X_train, y_train)

    # Save model
    builder.save_model('models/best_home_price_model.pkl')

    # 4. Evaluation
    logger.info("Step 4: Evaluation and Explainability")
    evaluator = ModelEvaluator()

    # Comprehensive evaluation
    evaluator.comprehensive_evaluation(
        final_model, X_train, X_test, y_train, y_test,
        model_name="Home_Price_Model", results_dir="results/"
    )

    # Predict future prices (next 5 years)
    logger.info("Predicting future prices")
    last_date = cleaned_data['date'].max()
    future_dates = pd.date_range(start=last_date, periods=61, freq='ME')[1:]  # Next 5 years monthly
    future_time_index = (future_dates - cleaned_data['date'].min()).days
    future_X = pd.DataFrame({'time_index': future_time_index})

    future_predictions = final_model.predict(future_X)

    future_df = pd.DataFrame({
        'date': future_dates,
        'predicted_price': future_predictions
    })

    future_df.to_csv('results/future_price_predictions.csv', index=False)
    logger.info("Future price predictions saved to results/future_price_predictions.csv")

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()

    # Get predictions from best model
    y_pred = best_model.predict(X_test)

    # Calculate comprehensive metrics
    metrics = evaluator.calculate_metrics(y_test.values, y_pred, "Best Random Forest")

    # Generate evaluation report
    dataset_info = {
        'name': 'Boston Housing',
        'samples': len(data),
        'features': X.shape[1]
    }
    report = evaluator.generate_evaluation_report("Random Forest", dataset_info, metrics)
    print(report)

    # Create evaluation plots
    evaluator.plot_residuals_analysis(y_test.values, y_pred,
                                    save_path='../results/residual_analysis.png')

    # Feature importance (for Random Forest)
    feature_names = X.columns.tolist()
    importance_scores = best_model.named_steps['regressor'].feature_importances_
    evaluator.plot_feature_importance(feature_names, importance_scores,
                                    save_path='../results/feature_importance.png')

    # Learning curves
    evaluator.plot_learning_curves(best_model, X_train, y_train,
                                 save_path='../results/learning_curves.png')

    # Model comparison
    evaluator.compare_models(results, save_path='../results/model_comparison.png')

    logger.info("Pipeline completed successfully!")
    logger.info("Check the 'results' directory for evaluation plots and reports")

if __name__ == "__main__":
    main()