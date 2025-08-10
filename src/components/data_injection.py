import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import logging

from src.components.data_transformation import DataTransformation

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # ✅ Use forward slashes to avoid invalid escape sequence issue
            df = pd.read_csv(r"notebook/data/StudentsPerformance.csv")
            logging.info("Read the dataset as DataFrame")

            # ✅ Ensure artifact folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise e

if __name__ == "__main__":
    # Step 1: Data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Data transformation
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
