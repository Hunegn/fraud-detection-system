import pandas as pd
import numpy as np
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FraudModelTrainer:
    def __init__(self, data_path, target_col):
        """
        Initializes the trainer with dataset path and target column.
        Args:
            data_path (str): Path to the CSV data file.
            target_col (str): Name of the target column (e.g., 'class' for Fraud_Data or 'Class' for creditcard).
        """
        self.data_path = data_path
        self.target_col = target_col
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models = {}
        logging.info("FraudModelTrainer initialized.")

    def load_data(self):
        """Load dataset from CSV file."""
        logging.info("Loading dataset...")
        self.data = pd.read_csv(self.data_path)
        logging.info(f"Data loaded with shape: {self.data.shape}")

    def prepare_data(self):
        """Separate features and target, then split data into training and testing sets."""
        logging.info("Preparing data...")
        # Drop non-numeric columns from features (adjust as needed) 
        X = self.data.drop(columns=[self.target_col]).select_dtypes(include=[np.number])
        y = self.data[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        logging.info("Train-test split completed.")

    def train_model(self, model, model_name, param_grid=None):
        """
        Train the provided model with optional hyperparameter tuning.
        Args:
            model: An instance of a scikit-learn model.
            model_name (str): A descriptive name for the model.
            param_grid (dict, optional): Parameter grid for GridSearchCV.
        Returns:
            Trained model.
        """
        logging.info(f"Training {model_name}...")
        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", error_score='raise')
            grid.fit(self.X_train, self.y_train)
            best_model = grid.best_estimator_
            logging.info(f"Best parameters for {model_name}: {grid.best_params_}")
        else:
            model.fit(self.X_train, self.y_train)
            best_model = model
        
        self.models[model_name] = best_model
        
        y_pred = best_model.predict(self.X_test)
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(self.X_test)[:, 1]
            roc_auc = roc_auc_score(self.y_test, y_prob)
        else:
            roc_auc = None

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        logging.info(f"{model_name} - Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}, ROC-AUC: {roc_auc}")
        print(f"\n{model_name} Evaluation:\n", classification_report(self.y_test, y_pred))
        mlflow.sklearn.log_model(best_model, model_name)
        
        return best_model

    def run_training(self):
        """Execute the entire training pipeline with multiple models."""
        self.load_data()
        self.prepare_data()
        
        with mlflow.start_run():
            mlflow.log_param('target_column', self.target_col)
            mlflow.log_param('test_size', 0.3)
            
            # Logistic Regression
            lr = LogisticRegression(max_iter=1000)
            self.train_model(lr, "Logistic Regression", param_grid={"C": [0.1, 1, 10]})
            
            # Decision Tree
            dt = DecisionTreeClassifier(random_state=42)
            self.train_model(dt, "Decision Tree", param_grid={"max_depth": [5, 10, 15]})
            
            # Random Forest
            rf = RandomForestClassifier(random_state=42)
            self.train_model(rf, "Random Forest", param_grid={"n_estimators": [50, 100], "max_depth": [5, 10]})
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(random_state=42)
            self.train_model(gb, "Gradient Boosting", param_grid={"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]})
            
            mlflow.end_run()

if __name__ == "__main__":
  
    trainer = FraudModelTrainer(data_path="../data/raw/Fraud_Data.csv", target_col="class")
    trainer.run_training()
