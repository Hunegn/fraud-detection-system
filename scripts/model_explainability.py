import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import logging


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelExplainability:
    def __init__(self, model_path, test_data_path, target_col):
        """
        Initialize the explainability module.
        Args:
            model_path (str): Path to the serialized trained model (e.g., fraud_model.pkl).
            test_data_path (str): Path to the CSV file containing the test data.
            target_col (str): Name of the target column.
        """
        self.model = joblib.load(model_path)
        logging.info(f"Loaded model from {model_path}.")
        self.data = pd.read_csv(test_data_path)
        logging.info(f"Loaded test data with shape: {self.data.shape}.")
        self.target_col = target_col
        # Assume features are all columns except the target. Adjust if needed.
        self.X_test = self.data.drop(columns=[self.target_col])
        self.y_test = self.data[self.target_col]
    
    def explain_with_shap(self):
        logging.info("Generating SHAP explanations...")
        
        # Use only numeric columns for SHAP
        numeric_X_test = self.X_test.select_dtypes(include=[np.number])
        logging.info("Numeric X_test shape: " + str(numeric_X_test.shape))
        
        # Create the SHAP explainer; disable additivity check
        explainer = shap.Explainer(self.model, numeric_X_test)
        shap_values = explainer(numeric_X_test, check_additivity=False)
        
        # Summary Plot: Provide feature names explicitly
        shap.summary_plot(shap_values, numeric_X_test, feature_names=numeric_X_test.columns, show=False)
        summary_filename = os.path.join("plots", "shap_summary_plot.png")
        plt.savefig(summary_filename)
        plt.close()
        logging.info(f"SHAP summary plot saved as {summary_filename}")
        
        # Force Plot for the first instance: Provide feature names explicitly
        shap.force_plot(
            explainer.expected_value, 
            shap_values.values[0, :], 
            numeric_X_test.iloc[0, :],
            feature_names=numeric_X_test.columns,
            matplotlib=True,
            show=False
        )
        force_filename = os.path.join("plots", "shap_force_plot.png")
        plt.savefig(force_filename)
        plt.close()
        logging.info(f"SHAP force plot saved as {force_filename}")
        
        # Dependence Plot for a selected feature, e.g., 'purchase_value'
        if 'purchase_value' in numeric_X_test.columns:
            shap.dependence_plot(
                'purchase_value', 
                shap_values.values, 
                numeric_X_test,
                feature_names=numeric_X_test.columns,
                show=False
            )
            dep_filename = os.path.join("plots", "shap_dependence_plot.png")
            plt.savefig(dep_filename)
            plt.close()
            logging.info(f"SHAP dependence plot saved as {dep_filename}")
        else:
            logging.warning("Feature 'purchase_value' not found for SHAP dependence plot.")



    def explain_with_lime(self):
        """
        Generate a LIME explanation for a single instance.
        """
        logging.info("Generating LIME explanation...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X_test),
            feature_names=self.X_test.columns,
            class_names=["Non-Fraud", "Fraud"],
            mode='classification'
        )
        
        # Explain the first instance
        exp = explainer.explain_instance(
            data_row=self.X_test.iloc[0].values,
            predict_fn=self.model.predict_proba,
            num_features=10
        )
        lime_filename = os.path.join("plots", "lime_explanation.html")
        exp.save_to_file(lime_filename)
        logging.info(f"LIME explanation saved as {lime_filename}")
    
    def run_explainability(self):
        self.explain_with_shap()
        self.explain_with_lime()


