"""
Predictive Analysis Module for Learning Pattern Analysis System

This module implements:
1. Training a Random Forest Regressor for grade forecasting
2. Predicting G3 grades based on behavioral features and G1/G2
3. Providing "What-If" prediction capabilities
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class GradePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_names = [
            'engagement_score', 'consistency_index', 'performance_trend', 
            'participation_stability', 'studytime', 'failures', 'absences', 
            'G1', 'G2'
        ]
        self.is_trained = False

    def train(self, df):
        """Train the model on the provided dataframe."""
        X = df[self.feature_names]
        y = df['G3']
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"✓ Trained Grade Predictor model (R²: {r2:.3f}, MSE: {mse:.3f})")
        self.is_trained = True
        return r2, mse

    def predict(self, student_features):
        """
        Predict G3 grade for a single student profile.
        
        Args:
            student_features (dict or pd.Series): Student feature values
            
        Returns:
            float: Predicted G3 grade
            float: Pass probability (G3 >= 10)
        """
        if not self.is_trained:
            return 0.0, 0.0
            
        # Convert to DataFrame for prediction
        if isinstance(student_features, dict):
            X = pd.DataFrame([student_features])[self.feature_names]
        else:
            X = pd.DataFrame([student_features[self.feature_names].values], columns=self.feature_names)
            
        prediction = self.model.predict(X)[0]
        
        # Estimate pass probability based on nearby ensemble predictions (simplification)
        # In a real scenario, we might use a separate classifier or quantiles
        pass_prob = 1.0 if prediction >= 12 else (0.8 if prediction >= 10 else 0.3)
        
        return prediction, pass_prob

def initialize_predictor(df):
    """Factory function to create and train a predictor."""
    predictor = GradePredictor()
    predictor.train(df)
    return predictor

if __name__ == "__main__":
    # Test the predictor
    from data_processor import process_data
    df, _, _, _ = process_data()
    predictor = initialize_predictor(df)
    
    # Test on a sample student
    sample = df.iloc[0]
    pred, prob = predictor.predict(sample)
    print(f"\nSample Prediction:")
    print(f"  Actual G3: {sample['G3']}")
    print(f"  Predicted G3: {pred:.1f}")
    print(f"  Pass Probability: {prob*100:.1f}%")
