"""
Model Building Module for Regression Pipeline
Implements various regression models with hyperparameter tuning and cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBuilder:
    """Class for building and training regression models."""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_params = None

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            random_state: Random state for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def define_models(self) -> Dict[str, Any]:
        """Define regression models to train."""
        self.models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'Lasso': {
                'model': Lasso(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }
        return self.models

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best models and scores
        """
        best_models = {}
        for name, model_info in self.models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model_info['model'])
            ])

            param_grid = {f'regressor__{k}': v for k, v in model_info['params'].items()}

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            best_models[name] = {
                'model': grid_search.best_estimator_,
                'params': grid_search.best_params_,
                'score': -grid_search.best_score_  # Convert back to positive MSE
            }

            logger.info(f"{name} best params: {grid_search.best_params_}, MSE: {-grid_search.best_score_}")

        # Select best model
        best_name = min(best_models, key=lambda x: best_models[x]['score'])
        self.best_model = best_models[best_name]['model']
        self.best_params = best_models[best_name]['params']

        logger.info(f"Best model: {best_name}")
        return best_models

    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the best model on full training data.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained model
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run tune_hyperparameters first.")

        self.best_model.fit(X_train, y_train)
        logger.info("Trained final model.")
        return self.best_model

    def save_model(self, file_path: str):
        """Save the trained model to disk."""
        if self.best_model is None:
            raise ValueError("No model to save.")
        joblib.dump(self.best_model, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> Any:
        """Load a trained model from disk."""
        self.best_model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return self.best_model