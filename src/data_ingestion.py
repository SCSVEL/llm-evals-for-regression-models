"""
Data Ingestion Module for Regression Pipeline
Handles loading data from various CSV sources for LLM evaluation tasks.
"""

import pandas as pd
import requests
from io import StringIO
import logging
from typing import Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Class for handling data ingestion from CSV files."""

    def __init__(self):
        self.data = None

    def load_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a local CSV file.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame containing the loaded data
        """
        try:
            self.data = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded data from {file_path}. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    def load_from_url(self, url: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV URL.

        Args:
            url: URL of the CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame containing the loaded data
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.data = pd.read_csv(StringIO(response.text), **kwargs)
            logger.info(f"Successfully loaded data from {url}. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data from {url}: {e}")
            raise

    def load_zillow_home_prices(self, file_path: str, region_name: str = "State College, PA") -> pd.DataFrame:
        """
        Load and process Zillow Home Value Index data for a specific region.

        Args:
            file_path: Path to the Zillow CSV file
            region_name: Name of the region to filter (e.g., "State College, PA")

        Returns:
            DataFrame with columns: date, price
        """
        # Load the data
        df = self.load_from_csv(file_path)

        # Filter for the specific region
        region_data = df[df['RegionName'] == region_name]
        if region_data.empty:
            raise ValueError(f"Region '{region_name}' not found in the data.")

        # Melt the wide format to long format
        date_columns = [col for col in df.columns if col.startswith('20') or col.startswith('19')]
        melted = region_data.melt(id_vars=['RegionID', 'RegionName', 'RegionType', 'StateName'],
                                  value_vars=date_columns,
                                  var_name='date',
                                  value_name='price')

        # Convert date to datetime
        melted['date'] = pd.to_datetime(melted['date'], format='%Y-%m-%d')

        # Drop NaN prices
        melted = melted.dropna(subset=['price'])

        # Sort by date
        melted = melted.sort_values('date').reset_index(drop=True)

        # Select relevant columns
        processed_data = melted[['date', 'price']]

        logger.info(f"Processed data for {region_name}. Shape: {processed_data.shape}")
        return processed_data
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.data = pd.read_csv(StringIO(response.text), **kwargs)
            logger.info(f"Successfully loaded data from {url}. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data from {url}: {e}")
            raise

    def validate_data(self) -> dict:
        """
        Perform basic data validation.

        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_from_csv or load_from_url first.")

        validation = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum()
        }

        logger.info("Data validation completed")
        return validation

    def get_summary(self) -> pd.DataFrame:
        """
        Get basic statistical summary of the data.

        Returns:
            DataFrame with summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        return self.data.describe()

# Example usage
if __name__ == "__main__":
    ingestion = DataIngestion()
    # Load Boston Housing data as example
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    data = ingestion.load_from_url(url)
    validation = ingestion.validate_data()
    summary = ingestion.get_summary()

    print("Data shape:", validation['shape'])
    print("Columns:", validation['columns'])
    print("Summary statistics:")
    print(summary)