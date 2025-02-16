import pandas as pd
import numpy as np
import socket
import struct
import logging
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except Exception:
        return np.nan

def get_country(ip_int, ip_df):
    if pd.isnull(ip_int):
        return np.nan
    row = ip_df[(ip_df['lower_bound_ip'] <= ip_int) & (ip_df['upper_bound_ip'] >= ip_int)]
    if not row.empty:
        return row.iloc[0]['country']
    return np.nan

class FraudDataPreprocessor:
    def __init__(self, fraud_data_path, ip_data_path, output_path):
        self.fraud_data_path = fraud_data_path
        self.ip_data_path = ip_data_path
        self.output_path = output_path
        self.data = None
        self.ip_df = None
        logging.info("FraudDataPreprocessor initialized.")

    def load_data(self):
        logging.info("Loading Fraud_Data...")
        self.data = pd.read_csv(self.fraud_data_path)
        logging.info(f"Fraud_Data loaded with shape: {self.data.shape}")
        
        logging.info("Loading IP Address Data...")
        self.ip_df = pd.read_csv(self.ip_data_path)
        logging.info(f"IpAddress_to_Country loaded with shape: {self.ip_df.shape}")

    def clean_data(self):
        logging.info("Cleaning data...")
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        logging.info(f"Data shape after dropping duplicates: {self.data.shape}")

        # Convert time columns to datetime (assumes column names 'purchase_time' and 'signup_time')
        self.data['purchase_time'] = pd.to_datetime(self.data['purchase_time'], errors='coerce')
        self.data['signup_time'] = pd.to_datetime(self.data['signup_time'], errors='coerce')

        # Handle missing values: Impute numeric with median; categorical with 'Unknown'
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna("Unknown")
        
        logging.info("Data cleaning completed.")

    def ip_conversion_and_merge(self):
        logging.info("Converting IP addresses to integers...")
        self.data['ip_int'] = self.data['ip_address'].apply(ip_to_int)
        
        # Convert IP range bounds to integers in the ip_df
        self.ip_df['lower_bound_ip'] = self.ip_df['lower_bound_ip_address'].apply(ip_to_int)
        self.ip_df['upper_bound_ip'] = self.ip_df['upper_bound_ip_address'].apply(ip_to_int)
        
        logging.info("Merging Fraud_Data with IP-to-Country mapping...")
        self.data['country'] = self.data['ip_int'].apply(lambda x: get_country(x, self.ip_df))
    
    def exploratory_analysis(self):
        logging.info("Performing exploratory data analysis...")
        
        # Log summary statistics for numeric columns only
        numeric_data = self.data.select_dtypes(include=[np.number])
        logging.info("Summary statistics (numeric):\n" + str(numeric_data.describe()))
        
        # Compute correlation only on numeric data
        corr = numeric_data.corr()
        logging.info("Correlation matrix:\n" + str(corr))

    
    def feature_engineering(self):
        logging.info("Performing feature engineering...")
        # Transaction Frequency: Count transactions per user
        freq_df = self.data.groupby("user_id")["purchase_value"].count().reset_index()
        freq_df.columns = ["user_id", "transaction_count"]
        self.data = self.data.merge(freq_df, on="user_id", how="left")
        
        # Transaction Velocity: Average time difference (in hours) between transactions per user
        self.data.sort_values(by=["user_id", "purchase_time"], inplace=True)
        self.data['time_diff_hours'] = self.data.groupby("user_id")['purchase_time'].diff().dt.total_seconds() / 3600.0
        velocity_df = self.data.groupby("user_id")['time_diff_hours'].mean().reset_index()
        velocity_df.columns = ["user_id", "avg_time_diff_hours"]
        self.data = self.data.merge(velocity_df, on="user_id", how="left")
        
        # Time-Based Features: Extract hour of day and day of week from purchase_time
        self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
        self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek
        
        logging.info("Feature engineering completed.")
    
    def normalize_and_scale(self):
        logging.info("Normalizing numerical features...")
        cols_to_scale = ['purchase_value', 'age', 'transaction_count', 'avg_time_diff_hours']
        scaler = StandardScaler()
        self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
        logging.info("Normalization and scaling completed.")
    
    def encode_categorical_features(self):
        logging.info("Encoding categorical features using one-hot encoding...")
        
        self.data = pd.get_dummies(self.data, columns=['source', 'browser', 'sex'], drop_first=True)
        logging.info("Categorical encoding completed.")
    
    def save_preprocessed_data(self):
        self.data.to_csv(self.output_path, index=False)
        logging.info(f"Preprocessed data saved to {self.output_path}.")

    def run_pipeline(self):
        self.load_data()
        self.clean_data()
        self.ip_conversion_and_merge()
        self.exploratory_analysis()
        self.feature_engineering()
        self.normalize_and_scale()
        self.encode_categorical_features()
        self.save_preprocessed_data()

if __name__ == "__main__":
    fraud_data_path = "../data/raw/Fraud_Data.csv"
    ip_data_path = "../data/raw/IpAddress_to_Country.csv"
    output_path = "../data/processed/processed_fraud_data.csv"
    
    preprocessor = FraudDataPreprocessor(fraud_data_path, ip_data_path, output_path)
    preprocessor.run_pipeline()
