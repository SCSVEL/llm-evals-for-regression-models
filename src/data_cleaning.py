"""
Data Cleaning Module for Regression Pipeline
Handles data preprocessing, missing values, outliers, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Class for comprehensive data cleaning operations."""

    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.encoders = {}

    def load_data(self, data: pd.DataFrame):
        """Load data for cleaning."""
        self.data = data.copy()
        logger.info(f"Loaded data for cleaning. Shape: {self.data.shape}")

    def handle_missing_values(self, strategy: str = 'interpolate', columns: Optional[list] = None) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.

        Args:
            strategy: 'drop', 'mean', 'median', 'mode', or 'interpolate'
            columns: Specific columns to handle. If None, handles all columns.

        Returns:
            DataFrame with missing values handled
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        data = self.data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns

        for col in columns:
            if data[col].isnull().sum() > 0:
                if strategy == 'drop':
                    data = data.dropna(subset=[col])
                elif strategy == 'mean':
                    data[col] = data[col].fillna(data[col].mean())
                elif strategy == 'median':
                    data[col] = data[col].fillna(data[col].median())
                elif strategy == 'mode':
                    data[col] = data[col].fillna(data[col].mode().iloc[0])
                elif strategy == 'interpolate':
                    data[col] = data[col].interpolate(method='linear')
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"Handled missing values using {strategy} strategy.")
        return data

    def remove_outliers(self, method: str = 'iqr', columns: Optional[list] = None, threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using specified method.

        Args:
            method: 'iqr' or 'zscore'
            columns: Columns to check for outliers
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        data = self.data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns

        for col in columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < threshold]
            else:
                raise ValueError(f"Unknown method: {method}")

        logger.info(f"Removed outliers using {method} method. New shape: {data.shape}")
        return data

    def feature_engineering(self) -> pd.DataFrame:
        """
        Perform feature engineering for time series regression.

        Returns:
            DataFrame with new features
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        data = self.data.copy()

        # Ensure date column is datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['year'] = data['date'].dt.year
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            data['year_month'] = data['date'].dt.to_period('M').astype(str)
            # Add time index for regression
            data['time_index'] = (data['date'] - data['date'].min()).dt.days

        logger.info("Performed feature engineering.")
        return data

    def scale_features(self, columns: list) -> Tuple[pd.DataFrame, dict]:
        """
        Scale numerical features.

        Args:
            columns: Columns to scale

        Returns:
            Tuple of scaled DataFrame and scaler parameters
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        data = self.data.copy()
        data[columns] = self.scaler.fit_transform(data[columns])
        scaler_params = {'mean': self.scaler.mean_, 'scale': self.scaler.scale_}
        logger.info(f"Scaled features: {columns}")
        return data, scaler_params

    def clean_data(self) -> pd.DataFrame:
        """
        Complete cleaning pipeline.

        Returns:
            Cleaned DataFrame
        """
        self.data = self.handle_missing_values()
        self.data = self.remove_outliers()
        self.data = self.feature_engineering()
        logger.info("Completed data cleaning pipeline.")
        return self.data

        logger.info(f"Handled missing values using {strategy} strategy")
        self.data = data
        return data

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows."""
        if self.data is None:
            raise ValueError("No data loaded.")

        initial_shape = self.data.shape
        self.data = self.data.drop_duplicates()
        final_shape = self.data.shape

        logger.info(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows")
        return self.data

    def detect_outliers_iqr(self, column: str, factor: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using IQR method.

        Args:
            column: Column name to check for outliers
            factor: IQR multiplier for outlier detection

        Returns:
            DataFrame containing outlier rows
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        logger.info(f"Detected {len(outliers)} outliers in column {column}")
        return outliers

    def remove_outliers_iqr(self, column: str, factor: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method.

        Args:
            column: Column name to remove outliers from
            factor: IQR multiplier

        Returns:
            DataFrame with outliers removed
        """
        outliers = self.detect_outliers_iqr(column, factor)
        self.data = self.data.drop(outliers.index)
        logger.info(f"Removed {len(outliers)} outliers from column {column}")
        return self.data

    def encode_categorical(self, columns: list) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.

        Args:
            columns: List of categorical columns to encode

        Returns:
            DataFrame with encoded columns
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        for col in columns:
            if col in self.data.columns:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])
                self.encoders[col] = encoder

        logger.info(f"Encoded categorical columns: {columns}")
        return self.data

    def scale_features(self, columns: Optional[list] = None) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Scale numerical features using StandardScaler.

        Args:
            columns: Columns to scale. If None, scales all numerical columns.

        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        scaled_data = self.data.copy()
        scaled_data[columns] = self.scaler.fit_transform(self.data[columns])

        logger.info(f"Scaled features: {columns}")
        return scaled_data, self.scaler

    def get_clean_summary(self) -> dict:
        """Get summary of cleaning operations performed."""
        if self.data is None:
            raise ValueError("No data loaded.")

        return {
            'final_shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum()
        }

# Example usage
if __name__ == "__main__":
    # Example with sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 100],  # outlier
        'feature2': [1.0, 2.0, np.nan, 4.0, 5.0],  # missing value
        'target': [1, 2, 3, 4, 5]
    })

    cleaner = DataCleaner()
    cleaner.load_data(data)

    # Handle missing values
    cleaner.handle_missing_values(strategy='mean')

    # Remove outliers
    outliers = cleaner.detect_outliers_iqr('feature1')
    print(f"Outliers detected: {len(outliers)}")

    # Get summary
    summary = cleaner.get_clean_summary()
    print("Clean data summary:", summary)